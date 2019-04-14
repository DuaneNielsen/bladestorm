import ray
import pprint
from argparse import ArgumentParser

@ray.remote
def get_python_env():
    import sys
    import ray.services
    return sys.version, ray.services.get_node_ip_address()


if __name__ == '__main__':
    parser = ArgumentParser(description='list connected workers')
    parser.add_argument('--redis-addr', default='localhost')
    parser.add_argument('--redis-port', default='6379')
    args = parser.parse_args()
    redis_addr = args.redis_addr + ":" + args.redis_port
    ray.init(redis_address=redis_addr)
    print(ray.get(get_python_env.remote()))