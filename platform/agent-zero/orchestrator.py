"""
Agent Orchestrator

Manages the lifecycle and communication of all agents in the system.
"""

from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging

from .base_agent import (
    BaseAgent,
    AgentState,
    AgentMessage,
    MessageType,
    AgentPriority
)


class OrchestratorState(Enum):
    """State of the orchestrator."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class BroadcastSubscription:
    """Subscription to a broadcast topic."""
    topic: str
    agent_id: str
    filter_func: Optional[Callable[[AgentMessage], bool]] = None


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    name: str
    agent_id: str
    action: str
    payload_transform: Optional[Callable[[Dict], Dict]] = None
    condition: Optional[Callable[[Dict], bool]] = None


@dataclass
class Workflow:
    """A workflow definition."""
    name: str
    steps: List[WorkflowStep]
    on_success: Optional[Callable[[Dict], None]] = None
    on_failure: Optional[Callable[[Exception], None]] = None


class AgentOrchestrator:
    """
    Central orchestrator for managing agents.

    Provides:
    - Agent lifecycle management
    - Message routing
    - Broadcast/subscription patterns
    - Workflow execution
    - Load balancing
    """

    def __init__(self, name: str = "StripeOrchestrator"):
        self.name = name
        self.state = OrchestratorState.INITIALIZING
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.subscriptions: Dict[str, List[BroadcastSubscription]] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.logger = logging.getLogger(f"orchestrator.{name}")
        self.created_at = datetime.now()
        self._running = False
        self._message_router_task: Optional[asyncio.Task] = None

        # Metrics
        self.messages_routed = 0
        self.broadcasts_sent = 0
        self.workflows_executed = 0

    async def start(self):
        """Start the orchestrator and all registered agents."""
        self.logger.info(f"Starting orchestrator: {self.name}")
        self._running = True
        self.state = OrchestratorState.RUNNING

        # Start message router
        self._message_router_task = asyncio.create_task(
            self._message_router()
        )

        # Initialize and start all agents
        for agent_id, agent in self.agents.items():
            try:
                if await agent.initialize():
                    self.agent_tasks[agent_id] = asyncio.create_task(
                        agent.run()
                    )
                    self.logger.info(f"Started agent: {agent.name}")
                else:
                    self.logger.error(f"Failed to initialize agent: {agent.name}")
            except Exception as e:
                self.logger.error(f"Error starting agent {agent.name}: {e}")

    async def stop(self):
        """Stop the orchestrator and all agents."""
        self.logger.info(f"Stopping orchestrator: {self.name}")
        self.state = OrchestratorState.SHUTTING_DOWN
        self._running = False

        # Stop all agents
        for agent_id, agent in self.agents.items():
            try:
                await agent.stop()
                if agent_id in self.agent_tasks:
                    self.agent_tasks[agent_id].cancel()
            except Exception as e:
                self.logger.error(f"Error stopping agent {agent.name}: {e}")

        # Stop message router
        if self._message_router_task:
            self._message_router_task.cancel()

        self.state = OrchestratorState.STOPPED
        self.logger.info(f"Orchestrator stopped: {self.name}")

    def register_agent(self, agent: BaseAgent):
        """
        Register an agent with the orchestrator.

        Args:
            agent: The agent to register
        """
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")

    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the orchestrator.

        Args:
            agent_id: The agent ID to unregister
        """
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            self.logger.info(f"Unregistered agent: {agent.name}")

            # Cancel the agent task
            if agent_id in self.agent_tasks:
                self.agent_tasks[agent_id].cancel()
                del self.agent_tasks[agent_id]

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)

    def get_agents_by_capability(self, capability: str) -> List[BaseAgent]:
        """Get all agents with a specific capability."""
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]

    async def send_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """
        Send a message to an agent.

        Args:
            message: The message to send

        Returns:
            Response message if synchronous
        """
        await self.message_queue.put(message)
        self.messages_routed += 1
        return None

    async def _message_router(self):
        """Background task for routing messages."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=0.5
                )

                # Route based on message type
                if message.message_type == MessageType.BROADCAST:
                    await self._handle_broadcast(message)
                else:
                    await self._route_message(message)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message router: {e}")

    async def _route_message(self, message: AgentMessage):
        """Route a message to its recipient."""
        recipient_id = message.recipient_id

        if recipient_id in self.agents:
            agent = self.agents[recipient_id]
            if agent.state == AgentState.RUNNING:
                await agent.receive_message(message)
            else:
                self.logger.warning(
                    f"Agent {agent.name} not running, message dropped"
                )
        else:
            self.logger.warning(f"Unknown recipient: {recipient_id}")

    async def _handle_broadcast(self, message: AgentMessage):
        """Handle a broadcast message."""
        topic = message.action

        if topic in self.subscriptions:
            for subscription in self.subscriptions[topic]:
                # Apply filter if any
                if subscription.filter_func:
                    if not subscription.filter_func(message):
                        continue

                agent = self.agents.get(subscription.agent_id)
                if agent and agent.state == AgentState.RUNNING:
                    await agent.receive_message(message)

        self.broadcasts_sent += 1

    def subscribe(
        self,
        topic: str,
        agent_id: str,
        filter_func: Optional[Callable[[AgentMessage], bool]] = None
    ):
        """
        Subscribe an agent to a broadcast topic.

        Args:
            topic: The topic to subscribe to
            agent_id: The agent ID
            filter_func: Optional filter function
        """
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []

        subscription = BroadcastSubscription(
            topic=topic,
            agent_id=agent_id,
            filter_func=filter_func
        )
        self.subscriptions[topic].append(subscription)
        self.logger.debug(f"Agent {agent_id} subscribed to {topic}")

    def unsubscribe(self, topic: str, agent_id: str):
        """
        Unsubscribe an agent from a topic.

        Args:
            topic: The topic
            agent_id: The agent ID
        """
        if topic in self.subscriptions:
            self.subscriptions[topic] = [
                s for s in self.subscriptions[topic]
                if s.agent_id != agent_id
            ]

    async def broadcast(
        self,
        topic: str,
        payload: Dict[str, Any],
        sender_id: str = "orchestrator"
    ):
        """
        Broadcast a message to all subscribers.

        Args:
            topic: The topic to broadcast on
            payload: The message payload
            sender_id: The sender ID
        """
        message = AgentMessage(
            sender_id=sender_id,
            recipient_id="*",
            message_type=MessageType.BROADCAST,
            action=topic,
            payload=payload
        )
        await self.message_queue.put(message)

    def register_workflow(self, workflow: Workflow):
        """Register a workflow."""
        self.workflows[workflow.name] = workflow
        self.logger.info(f"Registered workflow: {workflow.name}")

    async def execute_workflow(
        self,
        workflow_name: str,
        initial_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a registered workflow.

        Args:
            workflow_name: Name of the workflow to execute
            initial_payload: Initial payload for the workflow

        Returns:
            Final payload after all steps
        """
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_name}")

        self.logger.info(f"Executing workflow: {workflow_name}")
        payload = initial_payload.copy()

        try:
            for step in workflow.steps:
                # Check condition if any
                if step.condition and not step.condition(payload):
                    self.logger.debug(
                        f"Skipping step {step.name} (condition not met)"
                    )
                    continue

                # Transform payload if needed
                step_payload = payload
                if step.payload_transform:
                    step_payload = step.payload_transform(payload)

                # Send message to agent
                message = AgentMessage(
                    sender_id="orchestrator",
                    recipient_id=step.agent_id,
                    message_type=MessageType.REQUEST,
                    action=step.action,
                    payload=step_payload,
                    priority=AgentPriority.HIGH
                )

                agent = self.agents.get(step.agent_id)
                if not agent:
                    raise ValueError(f"Agent not found: {step.agent_id}")

                # Execute step
                response = await agent.process_message(message)
                if response and response.payload:
                    payload.update(response.payload)

            self.workflows_executed += 1

            if workflow.on_success:
                workflow.on_success(payload)

            return payload

        except Exception as e:
            self.logger.error(f"Workflow {workflow_name} failed: {e}")
            if workflow.on_failure:
                workflow.on_failure(e)
            raise

    async def request_capability(
        self,
        capability: str,
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Request a capability from any agent that provides it.

        Args:
            capability: The capability to request
            payload: The request payload
            timeout: Timeout in seconds

        Returns:
            Response message or None
        """
        agents = self.get_agents_by_capability(capability)
        if not agents:
            self.logger.warning(f"No agent provides capability: {capability}")
            return None

        # Simple load balancing: pick agent with fewest pending messages
        agent = min(agents, key=lambda a: a.message_queue.qsize())

        message = AgentMessage(
            sender_id="orchestrator",
            recipient_id=agent.agent_id,
            message_type=MessageType.REQUEST,
            action=capability,
            payload=payload
        )

        return await agent.process_message(message)

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        agent_statuses = {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }

        return {
            'name': self.name,
            'state': self.state.value,
            'agents_count': len(self.agents),
            'agents': agent_statuses,
            'subscriptions': {
                topic: len(subs) for topic, subs in self.subscriptions.items()
            },
            'workflows': list(self.workflows.keys()),
            'metrics': {
                'messages_routed': self.messages_routed,
                'broadcasts_sent': self.broadcasts_sent,
                'workflows_executed': self.workflows_executed
            },
            'created_at': self.created_at.isoformat()
        }
