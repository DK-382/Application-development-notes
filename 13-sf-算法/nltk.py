import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# 定义智能体类
class Agent:
    def __init__(self, name):
        self.name = name
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, message):
        """
        分析消息的情感倾向
        :param message: 输入的消息
        :return: 情感分析结果
        """
        return self.sia.polarity_scores(message)

    def respond(self, message):
        """
        根据消息的情感倾向生成响应
        :param message: 输入的消息
        :return: 智能体的响应
        """
        sentiment = self.analyze_sentiment(message)
        if sentiment['compound'] > 0.5:
            return f"{self.name}: You seem happy! I'm glad to hear that."
        elif sentiment['compound'] < -0.5:
            return f"{self.name}: You seem upset. Is there something I can help with?"
        else:
            return f"{self.name}: I'm not sure how you feel about that."

# 定义Swarm类，用于协调多个智能体
class Swarm:
    def __init__(self):
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def run(self, message):
        responses = []
        for agent in self.agents:
            response = agent.respond(message)
            responses.append(response)
        return responses

# 创建Swarm实例
swarm = Swarm()

# 创建两个智能体并添加到Swarm中
agent_a = Agent(name="Agent A")
agent_b = Agent(name="Agent B")

swarm.add_agent(agent_a)
swarm.add_agent(agent_b)

# 运行Swarm并发送消息
messages = ["I am so excited about the new project!", "I am really sad about the loss."]
responses = swarm.run(messages[0])

# 打印智能体的响应
for response in responses:
    print(response)