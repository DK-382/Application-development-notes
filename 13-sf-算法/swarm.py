# 定义智能体类
class Agent:
    def __init__(self, name, instructions):
        """
        初始化智能体
        :param name: 智能体的名称
        :param instructions: 智能体的指令或行为规则
        """
        self.name = name
        self.instructions = instructions

    def respond(self, message):
        """
        根据智能体的指令生成响应
        :param message: 接收到的消息
        :return: 智能体的响应
        """
        # 这里可以添加更复杂的逻辑，目前简单地返回指令和消息
        return f"{self.name}: {self.instructions} - I received: {message}"

# 定义Swarm类，用于协调多个智能体
class Swarm:
    def __init__(self):
        """
        初始化Swarm，用于管理多个智能体
        """
        self.agents = []

    def add_agent(self, agent):
        """
        添加智能体到Swarm
        :param agent: 要添加的智能体
        """
        self.agents.append(agent)

    def run(self, message):
        """
        运行Swarm，让所有智能体处理消息
        :param message: 要处理的消息
        :return: 所有智能体的响应列表
        """
        responses = []
        for agent in self.agents:
            response = agent.respond(message)
            responses.append(response)
        return responses

# 创建Swarm实例
swarm = Swarm()

# 创建两个智能体并添加到Swarm中
agent_a = Agent(name="Agent A", instructions="You are a helpful agent.")
agent_b = Agent(name="Agent B", instructions="Only speak in Haikus.")

swarm.add_agent(agent_a)
swarm.add_agent(agent_b)

# 运行Swarm并发送消息
messages = ["Hello, I need assistance.", "What is the weather like today?"]
responses = swarm.run(messages[0])

# 打印智能体的响应
for response in responses:
    print(response)

# 如果你想要一个更复杂的交互，你可以在Agent类中定义更复杂的逻辑
# 例如，agent_b可以只在特定消息上响应