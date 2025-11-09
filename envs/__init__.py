# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.


from gymnasium.envs.registration import register

# Register the DM Control environments.
from dm_control import suite

# Custom DM Control domains can be registered as follows:
# from . import <custom dm_env module>
# assert hasattr(<custom dm_env module>, 'SUITE')
# suite._DOMAINS['<custom dm_env module>'] = <custom dm_env module>

# Register all of the DM control tasks
for domain_name, task_name in suite._get_tasks(tag=None):
    # Import state domains
    ID = f'{domain_name.capitalize()}{task_name.capitalize()}-v0'
    register(id=ID, 
             entry_point='envs.dm_control:DMControlEnv', 
             kwargs={'domain_name': domain_name, 
                     'task_name': task_name,
                     'action_minimum': -1.0,
                     'action_maximum': 1.0,
                     'action_repeat': 1,
                     'from_pixels': False,
                     'flatten': True,
                     'stack': 1}, 
             )

    # Import vision domains as specified in DRQ-v2
    ID = f'{domain_name.capitalize()}{task_name.capitalize()}-vision-v0'
    camera_id = dict(quadruped=2).get(domain_name, 0)
    register(id=ID, 
             entry_point='envs.dm_control:DMControlEnv', 
             kwargs={'domain_name': domain_name, 
                     'task_name': task_name,
                     'action_repeat': 2,
                     'action_minimum': -1.0,
                     'action_maximum': 1.0,
                     'from_pixels': True,
                     'height': 84,
                     'width': 84,
                     'camera_id': camera_id,
                     'flatten': False,
                     'stack': 3}, 
             )
