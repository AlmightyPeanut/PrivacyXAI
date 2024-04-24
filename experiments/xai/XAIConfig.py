from dataclasses import dataclass, field


@dataclass
class XAIConfig:
    methods: set = field(default_factory=lambda: {})
