# coding: utf-8
# Copyright (c) 2025 inclusionAI.
from typing import List, Dict, Any

from aworld.core.exceptions import AWorldRuntimeException

from aworld.core.agent.base import AgentResult
from aworld.core.agent.swarm import Swarm
from aworld.core.model_output_parser import ModelOutputParser
from aworld.core.task import Task, TaskResponse
from aworld.utils.run_util import exec_tasks

from aworld.agents.llm_agent import Agent
from aworld.core.common import Observation, ActionModel, Config


class TaskAgent(Agent):
    """Support for swarm execution of in the hybrid nested swarm."""

    def __init__(self,
                 name: str,
                 swarm: Swarm,
                 conf: Config = None,
                 model_output_parser: ModelOutputParser[TaskResponse, AgentResult] = None,
                 **kwargs):
        super().__init__(name=name, conf=conf, model_output_parser=model_output_parser, **kwargs)
        self.swarm = swarm
        if not self.swarm:
            raise AWorldRuntimeException("no swarm in task agent.")

    def reset(self, options: Dict[str, Any] = None):
        super().reset(options)
        if not options:
            self.swarm.reset()
        else:
            self.swarm.reset(options.get("task"), options.get("context"), options.get("tools"))

    async def policy(self, observation: Observation, info: Dict[str, Any] = {}, **kwargs) -> List[ActionModel]:
        self._finished = False
        task = Task(input=observation.content, swarm=self.swarm)
        results = await exec_tasks([task])
        res = []
        for key, result in results.items():
            # result is TaskResponse
            if self.model_output_parser:
                # use output parser
                output = await self.model_output_parser.parse(result)
                res.extend(output.actions)
            else:
                if result.success:
                    info = result.answer
                else:
                    info = result.msg
                res.append(ActionModel(agent_name=self.name(), policy_info=info))

        self._finished = True
        return res
