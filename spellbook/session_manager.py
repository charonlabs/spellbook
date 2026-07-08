import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from spellbook.backends import build_backend
from spellbook.custom import CustomSurface
from spellbook.debug_visibility import (
    DebugEmitter,
    settings_from_runtime_config_records,
)
from spellbook.footer import (
    FooterController,
    FooterControllerRoundLifecycle,
)
from spellbook.fork import ForkConfig, ForkRunner, QuantumForkConfig
from spellbook.hearth import HearthSettings
from spellbook.ir_types import IRInboundMessage, IRSkillCatalog
from spellbook.loop import run_loop
from spellbook.nursery import Nursery
from spellbook.rehydrator import Rehydrator
from spellbook.session_lifecycle import (
    CompositeSessionLifecycle,
    SessionContext,
    SessionLifecycle,
)
from spellbook.slash_commands import SlashCommandHandler, parse_slash_command_message
from spellbook.skills.manager import SkillManager, SkillManagerRoundLifecycle
from spellbook.system_response import SystemResponse
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.timekeeper import (
    Timekeeper,
    TimekeeperRoundLifecycle,
    TimekeeperSessionLifecycle,
)
from spellbook.tools.common import ToolMetadata
from spellbook.tools.registry import ToolRegistry

from .cancel_token import CancelToken
from .config import SpellbookConfig
from .executor import Executor
from .generator import Generator
from .homunculus import Homunculus, HomunculusRoundLifecycle
from .inbound import InboundInjectionRoundLifecycle, InboundMessageQueue
from .recorder import Recorder, RecordingRoundLifecycle, RecordTap
from .round_lifecycle import CompositeRoundLifecycle, RoundLifecycle

SessionState = Literal["idle", "running", "dreaming", "suspended"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _RuntimeBreadcrumb:
    package_path: str
    git_sha: str
    git_state: str


class SessionBuilder(Protocol):
    async def __call__(
        self,
        transcript_path: Path,
        config: SpellbookConfig | None = None,
        lifecycle: SessionLifecycle | None = None,
        fork_config: ForkConfig | None = None,
        session_id: str | None = None,
        pre_round_lifecycle: RoundLifecycle | None = None,
        post_round_lifecycle: RoundLifecycle | None = None,
        record_tap: RecordTap | None = None,
        custom_surface: CustomSurface | None = None,
    ) -> "SessionManager": ...


class SessionManager:
    def __init__(
        self,
        *,
        session_id: str,
        inbound_queue: InboundMessageQueue,
        homunculus: Homunculus,
        generator: Generator,
        executor: Executor,
        round_lifecycle: RoundLifecycle,
        session_lifecycle: SessionLifecycle,
        recorder: Recorder,
        config: SpellbookConfig,
        tool_registry: ToolRegistry,
        transcript_path: Path,
        nursery: Nursery,
        skill_manager: SkillManager,
        debug_emitter: DebugEmitter | None = None,
        fork_config: ForkConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.inbound_queue = inbound_queue
        self.homunculus = homunculus
        self.generator = generator
        self.executor = executor
        self.round_lifecycle = round_lifecycle
        self.session_lifecycle = session_lifecycle
        self.recorder = recorder
        self.cancel_token: CancelToken | None = (
            None  # TODO: need to figure out the correct threading here
        )
        self.config = config
        self.tool_registry = tool_registry
        self.transcript_path = transcript_path
        self.nursery = nursery
        self.skill_manager = skill_manager
        self.fork_config = fork_config
        self.state: SessionState = "suspended"
        self._shutdown_requested = False
        self._ctx: SessionContext = SessionContext(
            session_id=session_id, turn_idx=recorder.current_turn_idx
        )
        self.debug_emitter = debug_emitter or DebugEmitter(
            recorder=recorder,
            session_lifecycle=session_lifecycle,
        )
        self.debug_emitter.bind_context(self._ctx)
        self.slash_commands = SlashCommandHandler(self)

    async def run(self) -> None:
        """Main entrypoint. Run until shutdown is requested."""
        while not self._shutdown_requested:
            await self._idle_phase()
            if not self._shutdown_requested:
                await self._running_phase()
        self.state = "suspended"
        await self.homunculus.shutdown()
        await self.debug_emitter.close()
        await self.session_lifecycle.on_shutdown(self._ctx)

    async def _shutdown_from_idle(self) -> None:
        await self.session_lifecycle.on_exit_idle(self._ctx, reason="shutdown")

    async def _idle_phase(self) -> None:
        self.state = "idle"
        await self.session_lifecycle.on_enter_idle(self._ctx)
        while True:
            msg = await self.inbound_queue.take_turn()
            if msg is None:
                await self._shutdown_from_idle()
                return
            if await self.handle_slash_command_message(msg) is not None:
                continue
            await self.session_lifecycle.on_exit_idle(self._ctx, reason="message")
            self.inbound_queue.push_back(msg)
            return

    async def _running_phase(self) -> None:
        self.state = "running"
        while self.inbound_queue.has_pending_turn() and not self._shutdown_requested:
            msg = await self.inbound_queue.take_turn()
            if msg is None:
                await self._shutdown_from_idle()
                return
            if await self.handle_slash_command_message(msg) is not None:
                continue
            turn_id = f"turn_{uuid4().hex}"
            self.recorder.start_turn(turn_id, msg.blocks)
            self._ctx.inbound = msg
            self._ctx.turn_idx = self.recorder.current_turn_idx
            await self.session_lifecycle.on_turn_started(self._ctx, turn_id)
            self.cancel_token = CancelToken()
            initial_blocks = await self.homunculus.render_context(msg.blocks)
            loop_result = await run_loop(
                generator=self.generator,
                executor=self.executor,
                lifecycle=self.round_lifecycle,
                initial_blocks=initial_blocks,
                cancel_token=self.cancel_token,
            )
            self.recorder.end_turn(loop_result.stop_reason)
            self.cancel_token = None
            await self.session_lifecycle.on_turn_ended(self._ctx, loop_result, turn_id)

    async def submit_message(self, msg: IRInboundMessage) -> SystemResponse | None:
        response = await self.handle_slash_command_message(msg)
        if response is not None:
            return response
        await self.inbound_queue.put(msg)
        return None

    def is_slash_command_message(self, msg: IRInboundMessage) -> bool:
        return parse_slash_command_message(msg) is not None

    async def handle_slash_command_message(
        self, msg: IRInboundMessage
    ) -> SystemResponse | None:
        parsed = parse_slash_command_message(msg)
        if parsed is None:
            return None
        command, args = parsed
        self._ctx.inbound = msg
        self._ctx.turn_idx = self.recorder.current_turn_idx
        response = await self.slash_commands.handle(command, args)
        self.recorder.write_system_response(response)
        await self.session_lifecycle.on_system_response(self._ctx, response)
        return response

    async def shutdown(self) -> None:
        self._shutdown_requested = True
        if self.cancel_token is not None:
            self.cancel_token.cancel()
        await self.inbound_queue.shutdown_queue()

    async def get_tool_meta(self) -> ToolMetadata:
        return self.executor.meta

    def interrupt(self) -> bool:
        if self.cancel_token is None or self.cancel_token.cancelled:
            return False
        self.cancel_token.cancel()
        return True

    @classmethod
    async def build(
        cls,
        transcript_path: Path,
        config: SpellbookConfig | None = None,
        lifecycle: SessionLifecycle | None = None,
        fork_config: ForkConfig | None = None,
        session_id: str | None = None,
        pre_round_lifecycle: RoundLifecycle | None = None,
        post_round_lifecycle: RoundLifecycle | None = None,
        record_tap: RecordTap | None = None,
        custom_surface: CustomSurface | None = None,
    ) -> "SessionManager":
        """Build a `SessionManager`. Always needs a transcript path. If not resuming (i.e. the transcript
        does not exist), must also pass a `SpellbookConfig` to intialize the new session"""
        is_resume = transcript_path.exists()
        if not is_resume:
            if config is None:
                raise ValueError(
                    "Transcript does not exist and no config was passed. Need one or the other."
                )
            profile = config.profile
            if session_id is None:
                session_id = f"{profile.session_id_prefix}_{uuid4().hex}"
            assert session_id is not None
            if profile.custom_surface:
                assert custom_surface is not None
            discover_skills = profile.should_discover_skills(
                custom_surface.include_tool_categories
                if custom_surface is not None
                else None
            )
            if discover_skills:
                initial_skill_manager = SkillManager(config=config)
                initial_catalog = initial_skill_manager.discover_skills()
                # TODO: make this go into its spot in the structured frame once that lands.
                # hacky append-to-sys-prompt for now.
                if initial_skill_manager.num_skills > 0:
                    config = config.model_copy(
                        update={
                            "system_prompt": config.system_prompt
                            + (f"\n\n{initial_skill_manager.render_prompt_addendum()}")
                        }
                    )
            else:
                initial_catalog = IRSkillCatalog()
            tool_registry = ToolRegistry.build(
                config.tool_categories,
                surface=profile.tool_surface,
                custom=custom_surface,
                include_quantum_submit=_include_quantum_submit(fork_config),
                body_url=config.body_url,
            )
            initial_recorder = Recorder(
                config=config,
                transcript_path=transcript_path,
                session_id=session_id,
                tool_registry=tool_registry,
                record_tap=record_tap,
            )

            initial_recorder.write_session_record(skill_catalog=initial_catalog)
        custom_tools = custom_surface.tools if custom_surface is not None else None
        rehydrator = Rehydrator(
            transcript_path=transcript_path, custom_tools=custom_tools
        )
        rehydrated = rehydrator.run()
        config = rehydrated.config
        profile = config.profile
        if profile.custom_surface:
            assert custom_surface is not None
        tool_registry = ToolRegistry.build(
            config.tool_categories,
            surface=profile.tool_surface,
            custom=custom_surface,
            include_quantum_submit=_include_quantum_submit(fork_config),
            body_url=config.body_url,
        )
        skill_manager = SkillManager(config=config)
        skill_manager.rehydrate(rehydrated)
        session_id = rehydrated.session_id
        breadcrumb = _spellbook_runtime_breadcrumb()
        logger.info(
            "spellbook.startup_breadcrumb session_id=%s resume=%s "
            "package_path=%s git_sha=%s git_state=%s",
            session_id,
            is_resume,
            breadcrumb.package_path,
            breadcrumb.git_sha,
            breadcrumb.git_state,
        )
        backend = build_backend(config)
        surface_builder = RequestSurfaceBuilder.from_config(
            backend=backend,
            config=config,
            tool_registry=tool_registry,
        )
        token_counter = backend.build_token_counter(
            config=config, surface_builder=surface_builder
        )
        recorder = Recorder(
            config=config,
            transcript_path=transcript_path,
            session_id=session_id,
            tool_registry=tool_registry,
            record_tap=record_tap,
        )
        inbound_queue = InboundMessageQueue()
        footer_controller = FooterController(
            inbound_queue=inbound_queue, recorder=recorder
        )
        footer_controller.rehydrate(rehydrated.pending_footers)
        recorder.set_state(
            turn_id=rehydrated.current_turn_id or "",
            turn=rehydrated.last_completed_turn
            if rehydrated.in_progress_turn is None
            else rehydrated.in_progress_turn,
            seq=(rehydrated.last_seq + 1) if rehydrated.last_seq is not None else 0,
        )
        timekeeper: Timekeeper | None = None
        if profile.ambient_time:
            timekeeper = Timekeeper(config=config, footer_c=footer_controller)
            if is_resume:
                timekeeper.note_resume(
                    previous_activity_time=Timekeeper.latest_activity_time(
                        rehydrated.records
                    ),
                    turn_idx=recorder.current_turn_idx,
                )
        session_lifecycle = lifecycle or SessionLifecycle()
        if profile.ambient_time:
            assert timekeeper is not None
            session_lifecycle = CompositeSessionLifecycle(
                [TimekeeperSessionLifecycle(timekeeper), session_lifecycle]
            )
        debug_emitter = DebugEmitter(
            recorder=recorder,
            session_lifecycle=session_lifecycle,
            settings=settings_from_runtime_config_records(
                rehydrated.runtime_config_updates
            ),
        )
        footer_controller.bind_debug_emitter(debug_emitter)
        fork_runner = ForkRunner(
            parent_config=config,
            parent_transcript_path=transcript_path,
            recorder=recorder,
            session_builder=cls.build,
            debug_emitter=debug_emitter,
        )
        nursery = Nursery(config=config, debug_emitter=debug_emitter)
        homunculus = Homunculus(
            config=config.hom_config,
            footer_c=footer_controller,
            recorder=recorder,
            token_counter=token_counter,
            nursery=nursery,
            fork_runner=fork_runner,
            fork_config=fork_config,
            debug_emitter=debug_emitter,
            hearth_settings=HearthSettings.from_config(config),
            enable_block_detection=profile.block_detection,
        )
        await homunculus.rehydrate(rehydrated)
        should_manage_skills = profile.should_discover_skills(
            custom_surface.include_tool_categories
            if custom_surface is not None
            else None
        )
        if profile.homunculus_lifecycle:
            lifecycles: list[RoundLifecycle] = [
                RecordingRoundLifecycle(recorder=recorder),
                HomunculusRoundLifecycle(homunculus=homunculus),
            ]
            if should_manage_skills:
                lifecycles.append(
                    SkillManagerRoundLifecycle(
                        footer_c=footer_controller,
                        manager=skill_manager,
                        recorder=recorder,
                    )
                )
            if profile.inbound_injection:
                lifecycles.append(
                    InboundInjectionRoundLifecycle(
                        inbound_queue=inbound_queue,
                        recorder=recorder,
                    )
                )
            if profile.ambient_time:
                assert timekeeper is not None
                lifecycles.append(TimekeeperRoundLifecycle(timekeeper))
            lifecycles.append(
                FooterControllerRoundLifecycle(
                    controller=footer_controller,
                    recorder=recorder,
                    debug_emitter=debug_emitter,
                )
            )
            comp_round_lifecycle = CompositeRoundLifecycle(lifecycles=lifecycles)
        else:
            comp_round_lifecycle = CompositeRoundLifecycle(
                lifecycles=[RecordingRoundLifecycle(recorder=recorder)]
            )
        if pre_round_lifecycle:
            comp_round_lifecycle.add_pre(pre_round_lifecycle)
        if post_round_lifecycle:
            comp_round_lifecycle.add_post(post_round_lifecycle)
        generator = Generator(
            backend=backend,
            config=config,
            surface_builder=surface_builder,
        )
        executor = Executor(
            config=config,
            transcript_path=transcript_path,
            skill_manager=skill_manager,
            homunculus=homunculus,
            registry=tool_registry,
            fork_config=fork_config,
        )
        return cls(
            session_id=session_id,
            inbound_queue=inbound_queue,
            homunculus=homunculus,
            generator=generator,
            executor=executor,
            round_lifecycle=comp_round_lifecycle,
            session_lifecycle=session_lifecycle,
            recorder=recorder,
            config=config,
            tool_registry=tool_registry,
            transcript_path=transcript_path,
            nursery=nursery,
            skill_manager=skill_manager,
            debug_emitter=debug_emitter,
            fork_config=fork_config,
        )


def _include_quantum_submit(fork_config: ForkConfig | None) -> bool:
    return not isinstance(fork_config, QuantumForkConfig) or fork_config.submit_tool


def _spellbook_runtime_breadcrumb() -> _RuntimeBreadcrumb:
    package_path = Path(__file__).resolve().parent
    repo_root = package_path.parent
    git_sha = _git_output(repo_root, ["rev-parse", "--short", "HEAD"]) or "unknown"
    git_status = _git_output(repo_root, ["status", "--short"])
    if git_status is None:
        git_state = "unknown"
    elif git_status:
        git_state = "dirty"
    else:
        git_state = "clean"
    return _RuntimeBreadcrumb(
        package_path=str(package_path),
        git_sha=git_sha,
        git_state=git_state,
    )


def _git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()
