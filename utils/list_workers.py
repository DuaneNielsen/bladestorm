import ray
import pprint
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='list connected workers')
    parser.add_argument('--redis-addr', default='localhost')
    parser.add_argument('--redis-port', default='6379')
    args = parser.parse_args()
    redis_addr = args.redis_addr + ":" + args.redis_port
    print(redis_addr)
    ray.init(redis_address=redis_addr)
    gsct = ray.global_state.client_table()
    pprint.pprint(gsct)