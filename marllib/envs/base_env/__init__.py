"""Local environment registry for MARLlib-style make_env calls."""

from Environment.multi_agent_dmp_rllib_env import MultiAgentDMPRllibEnv

ENV_REGISTRY = {
    "multi_agent_dmp": MultiAgentDMPRllibEnv,
}

