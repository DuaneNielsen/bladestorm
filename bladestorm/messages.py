import base64
import json
import pickle
from abc import ABC
import redis
import uuid
import threading


def encode(object):
    env_pickle = pickle.dumps(object, 0)
    return base64.b64encode(env_pickle).decode()


def decode(object):
    return pickle.loads(base64.b64decode(object.encode()))


class MessageDecoder:
    def __init__(self):
        self.lookup = {}
        self.register(RolloutMessage)
        self.register(EpisodeMessage)
        self.register(StopMessage)
        self.register(StopAllMessage)
        self.register(StoppedMessage)
        self.register(ResetMessage)
        self.register(TrainingProgress)
        self.register(KillMessage)

    def register(self, message_class):
        """ Registers a message class's decode in a lookup table"""
        self.lookup[message_class.header()] = message_class.decode

    def decode(self, message):
        d = json.loads(message)
        msg = d['msg']

        # lookup the decode method and pass it the message
        if msg in self.lookup:
            return self.lookup[msg](d)
        else:
            raise Exception


class Transport(ABC):
    def publish(self, queue, msg):
        pass

    def subscribe(self, queue):
        pass

    def listen(self, block=True):
        """
        Returns a list of messages
        :param block: to block or not
        :return: a list of messages
        """
        pass


class BaseTransport(Transport):
    def __init__(self):
        self.handler = MessageHandler()
        self.handler.register(KillMessage, None)
        self.decoder = MessageDecoder()


class RedisTransport(BaseTransport):
    def __init__(self, host='localhost', port=6379, db=0):
        super().__init__()
        self.r = redis.Redis(host, port, db)
        self.p = self.r.pubsub()

    def publish(self, channel, msg):
        data = msg.encode()
        self.r.publish(channel, data)

    def subscribe(self, channel):
        self.p.subscribe(channel)

    def processOne(self):
        message = self.p.get_message()
        if message['type'] == "message":
            self.handler.handle(message)

    def listen(self, block=True):
        for message in self.p.listen():
            if message['type'] == "message":
                msg = self.decoder.decode(message['data'])
                if msg.header() == 'KILL':
                    break
                else:
                    self.handler.handle(msg)


class Message:
    def __init__(self, server_uuid):
        self.content = None
        self._header = self.header()
        self.server_uuid = server_uuid
        self._header_content = f'"msg":"{self._header}", "server_uuid": "{self.server_uuid}"'

    def encode(self):
        self.content = f'{{{self._header_content}}}'
        return self.content

    @classmethod
    def header(cls):
        return cls.header()

    @classmethod
    def decode(cls, d):
        return cls(d['server_uuid'])


class KillMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'KILL'


class StopMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'STOP'


class StopAllMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'STOPALL'


class ResetMessage(Message):
    def __init__(self, server_uuid):
        super().__init__(server_uuid)

    @classmethod
    def header(cls):
        return 'RESET'


class StoppedMessage(Message):
    def __init__(self, server__uuid):
        super().__init__(server__uuid)

    @classmethod
    def header(cls):
        return 'STOPPED'


class TrainingProgress(Message):
    def __init__(self, server_uuid, steps):
        super().__init__(server_uuid)
        self.steps = steps

    @classmethod
    def header(cls):
        return 'training_progress'

    def encode(self):
        self.content = f'{{ {self._header_content}, "steps":{self.steps} }}'
        return self.content

    @classmethod
    def decode(cls, encoded):
        return TrainingProgress(encoded['server_uuid'], encoded['steps'])


class EpisodeMessage(Message):
    def __init__(self, server_uuid, id, steps, total_reward):
        super().__init__(server_uuid)
        self.id = id
        self.steps = int(steps)
        self.total_reward = float(total_reward)

    def encode(self):
        return f'{{ {self._header_content}, "id":"{self.id}", "steps":"{self.steps}", "total_reward":"{self.total_reward}"}}'

    @classmethod
    def header(cls):
        return 'episode'

    @classmethod
    def decode(cls, encoded):
        return EpisodeMessage(encoded['server_uuid'], encoded['id'], encoded['steps'], encoded['total_reward'])


class RolloutMessage(Message):
    def __init__(self, server_uuid, id, policy, env_config):
        super().__init__(server_uuid)
        self.policy = policy
        self.env_config = env_config
        self.id = int(id)

    def encode(self):
        env_pickle = encode(self.env_config)
        policy_pickle = encode(self.policy)
        self.content = f'{{ {self._header_content}, "id":"{self.id}", "policy":"{policy_pickle}", "env_config":"{env_pickle}" }}'
        return self.content

    @classmethod
    def header(cls):
        return 'rollout'

    @classmethod
    def decode(cls, d):
        server_uuid = d['server_uuid']
        id = d['id']
        policy = decode(d['policy'])
        env_config = decode(d['env_config'])
        return RolloutMessage(server_uuid, id, policy, env_config)


class MessageHandler:
    def __init__(self):
        self.decoder = MessageDecoder()
        self.handler = {}

    def register(self, msg, callback):
        self.handler[msg.header()] = callback

    def handle(self, msg):
        if msg.header() in self.handler:
            callback = self.handler[msg.header()]
            if callback is not None:
                callback(msg)


class Server(threading.Thread):
    def __init__(self, transport, channel):
        super().__init__()
        self.id = uuid.uuid4()
        self.t = transport
        self.channel = channel
        self.t.subscribe(channel)
        self.stopped = False
        self.t.handler.register(ResetMessage, self.reset)
        self.t.handler.register(StopAllMessage, self.stopAll)

    def register(self, msg_class, func):
        self.t.handler.register(msg_class, func)

    def run(self):
        self.t.listen()

    def reset(self, _):
        self.stopped = False

    def stopAll(self, msg):
        self.stopped = True
        self.t.publish(self.channel, StoppedMessage(self.id))