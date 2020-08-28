import ni_mon_client, ni_nfvo_client
from ni_mon_client.rest import ApiException
from ni_nfvo_client.rest import ApiException
from datetime import datetime, timedelta
from config import cfg
from torch_dqn import *

import numpy as np
import threading
import datetime as dt
import math
import os
import time
import subprocess
from pprint import pprint
import random

# Parameters
# OpenStack Parameters
openstack_network_id = "" # Insert OpenStack Network ID to be used for creating SFC

# <Important!!!!> parameters for Reinforcement Learning (DQN in this codes)
learning_rate = 0.01            # Learning rate
gamma         = 0.98            # Discount factor
buffer_limit  = 5000            # Maximum Buffer size
batch_size    = 32              # Batch size for mini-batch sampling
num_neurons = 128               # Number of neurons in each hidden layer
epsilon = 0.08                  # epsilon value of e-greedy algorithm
required_mem_size = 200         # Minimum number triggering sampling
print_interval = 20             # Number of iteration to print result during DQN

# Global values
sample_user_data = "#cloud-config\n password: %s\n chpasswd: { expire: False }\n ssh_pwauth: True\n manage_etc_hosts: true\n runcmd:\n - sysctl -w net.ipv4.ip_forward=1"
scaler_list = []

# get_monitoring_api(): get ni_monitoring_client api to interact with a monitoring module
# Input: null
# Output: monitoring moudle's client api
def get_monitoring_api():

    ni_mon_client_cfg = ni_mon_client.Configuration()
    ni_mon_client_cfg.host = cfg["ni_mon"]["host"]
    ni_mon_api = ni_mon_client.DefaultApi(ni_mon_client.ApiClient(ni_mon_client_cfg))

    return ni_mon_api


# get_nfvo_sfc_api(): get ni_nfvo_sfc api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfc api
def get_nfvo_sfc_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfc_api = ni_nfvo_client.SfcApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfc_api


# get_nfvo_sfcr_api(): get ni_nfvo_sfcr api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's sfcr api
def get_nfvo_sfcr_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()
    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_sfcr_api = ni_nfvo_client.SfcrApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_sfcr_api


# get_nfvo_vnf_api(): get ni_nfvo_vnf api to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf api
def get_nfvo_vnf_api():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_api = ni_nfvo_client.VnfApi(ni_nfvo_client.ApiClient(nfvo_client_cfg))

    return ni_nfvo_vnf_api


# get_nfvo_vnf_spec(): get ni_nfvo_vnf spec to interact with a nfvo module
# Input: null
# Output: nfvo moudle's vnf spec
def get_nfvo_vnf_spec():

    nfvo_client_cfg = ni_nfvo_client.Configuration()

    nfvo_client_cfg.host = cfg["ni_nfvo"]["host"]
    ni_nfvo_vnf_spec = ni_nfvo_client.VnfSpec(ni_nfvo_client.ApiClient(nfvo_client_cfg))
    ni_nfvo_vnf_spec.flavor_id = cfg["flavor"]["default"]
    ni_nfvo_vnf_spec.user_data = sample_user_data % cfg["instance"]["password"]

    return ni_nfvo_vnf_spec


# get_ip_from_vm(vm_id): get a data plane IP of VM instance
# Input: vm instance id
# Output: port IP of the data plane
def get_ip_from_id(vm_id):

    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instance(vm_id)

    ## Get ip address of specific network
    ports = query.ports
    network_id = openstack_network_id

    for port in ports:
        if port.network_id == network_id:
            return port.ip_addresses[-1]


# get_node_info(): get all node information placed in environment
# Input: null
# Output: Node information list
def get_node_info():
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_nodes()

    response = [ node_info for node_info in query if node_info.type == "compute"]

    return response


# get_vnf_info(sfc_prefix, sfc_vnfs): get each VNF instance information from monitoring module
# Input: Prefix of VNF instance name, SFC order tuple or list [example] ("client", "firewall", "dpi", "ids", "proxy")
# Output: VNF information list
def get_vnf_info(sfc_prefix, sfc_vnfs):

    # Get information of VNF instances which are used for SFC
    ni_mon_api = get_monitoring_api()
    query = ni_mon_api.get_vnf_instances()

    selected_vnfi = [ vnfi for vnfi in query for vnf_type in sfc_vnfs if vnfi.name.startswith(sfc_prefix + vnf_type) ]

    vnfi_list = []
    num_vnf_type = []
    temp = []

    # Sort VNF informations for creating states
    for vnf_type in sfc_vnfs:
        i =  sfc_vnfs.index(vnf_type)

        temp.append([])

        temp[i] = [ vnfi for vnfi in selected_vnfi if vnfi.name.startswith(sfc_prefix + vnf_type) ]
        temp[i].sort(key=lambda vnfi: vnfi.name)

        vnfi_list = vnfi_list + temp[i]

    return vnfi_list


# get_sfcr_by_name(sfcr_name): get sfcr information by using sfcr_name from NFVO module
# Input: sfcr name
# Output: sfcr_info
def get_sfcr_by_name(sfcr_name):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.name == sfcr_name ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info


# get_sfcr_by_id(sfcr_id): get sfc information by using sfcr_id from NFVO module
# Input: sfcr_id, FYI, sfcr is a flow classifier in OpenStack
# Output: sfcr_info
def get_sfcr_by_id(sfcr_id):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()
    query = ni_nfvo_sfcr_api.get_sfcrs()

    sfcr_info = [ sfcri for sfcri in query if sfcri.id == sfcr_id ]
    sfcr_info = sfcr_info[-1]

    return sfcr_info


# get_sfc_by_name(sfc_name): get sfc information by using sfc_name from NFVO module
# Input: sfc name
# Output: sfc_info
def get_sfc_by_name(sfc_name):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()
    query = ni_nfvo_sfc_api.get_sfcs()

    sfc_info = [ sfci for sfci in query if sfci.sfc_name == sfc_name ]

    if len(sfc_info) == 0:
        return False

    sfc_info = sfc_info[-1]

    return sfc_info


# get_tier_status(vnf_info, sfc_info): get each tier status, each tier includes same type VNF instances
# Input: vnf_info, sfc_info
# Output: each tier status showing CPU utilization (%), Memory utilization (%), Number of disk operations, distribution, size)
def get_tier_status(vnf_info, sfc_info):

    ni_mon_api = get_monitoring_api()
    resource_type = ["cpu_usage___value___gauge",
                     "memory_free___value___gauge",
                     "vda___disk_ops___read___derive",
                     "vda___disk_ops___write___derive"]

    vnf_instances_ids = sfc_info.vnf_instance_ids
    tier_status = []
    num_nodes = len(get_node_info())

    # Set time-period to get resources
    end_time = dt.datetime.now()
    start_time = end_time - dt.timedelta(seconds = 10)

    # Select each tier (VNF type)
    for tier_vnfs in vnf_instances_ids:
        tier_values = []
        tier_distribution = []

        # Select each instance ID in each tier
        for vnf in tier_vnfs:
            vnf_values = []

            for vnfi in vnf_info:
                if vnfi.id == vnf:
                    tier_distribution.append(vnfi.node_id)

            # Calculate resource status of each vnf
            for type in resource_type:

                query = ni_mon_api.get_measurement(vnf, type, start_time, end_time)
                value = 0
                memory_total = 0

                for response in query:
                    value = value + response.measurement_value

                num_query = len(query)
                value = 0 if num_query == 0 else value/num_query

                # Memory utilization calculation
                if type.startswith("memory"):
                    flavor_id = ni_mon_api.get_vnf_instance(vnf).flavor_id
                    memory_ram_mb = ni_mon_api.get_vnf_flavor(flavor_id).ram_mb
                    memory_total = 1000000 * memory_ram_mb

                    value = 100*(1-(value/memory_total)) if num_query != 0 else 0

                vnf_values.append(value) # CPU, Memory, Disk utilization of each VNF

            tier_values.append(vnf_values) # CPU, Memory, Disk utilization of each tier

        # Define status
        cpu, memory, disk = 0, 0, 0 # , network = 0, 0, 0, 0

        # Calculate sum of each resource utilization of each tier
        for vnf_values in tier_values:
            cpu = cpu + vnf_values[0]
            memory = memory + vnf_values[1]
            disk = disk + vnf_values[2] + vnf_values[3]

        # Calculate average resource utilization to define state
        tier_size = len(tier_values)
        tier_distribution = list(set(tier_distribution))

        status = {"cpu" : cpu/tier_size, "memory" : memory/tier_size, "disk" : disk/tier_size, "distribution" : len(tier_distribution)/num_nodes, "size" : tier_size }

        tier_status.append(status)

    return tier_status


# get_state(tier_status): pre-processing tier_status to make tensor as an input data
# Input: tier_status
# Output: np.array (tensor for input values)
def get_state(tier_status):
        s = []
        for tier in tier_status:
            s.append(tier["cpu"])
            s.append(tier["memory"])
            s.append(tier["disk"])
            s.append(tier["distribution"])
            s.append(tier["size"])

        return np.array(s)


# get_target_tier(tier_status, flag): get target tier to be applied auto-scaling action
# Input: tier_status, flag (flag is a value to check scaling in or out, positive number is scaling out)
# Output: tier index (if index is a negative number, no tier for scaling)
def get_target_tier(tier_status, flag):

    tier_scores = []

    for tier in tier_status:
        resource_utilization = 0.9*tier["cpu"]+0.1*tier["memory"]
        size = tier["size"]

        if flag > 0: # Add action # 분포도가 작은 놈을 (분포도 값이 큰놈 역수) 스케일 아웃
            scaling_mask = 0.0 if size > 4 else 1.0
            score = math.exp(resource_utilization)
            score = scaling_mask*(1/tier["distribution"])*score
        else: # Remove action # 분포도가 큰 놈을 스케일 인
            scaling_mask = 0.0 if size < 2 else 1.0
            score = math.exp(-resource_utilization)
            score = scaling_mask*tier["distribution"]*score

        tier_scores.append(score)

    # Target tier has the highest value
    high_score = max(tier_scores)

    if high_score == 0: # No tier for scaling
        return -1

    return tier_scores.index(max(tier_scores))


# deploy_vnf(vnf_spec): deploy VNF instance in OpenStack environment
# Input: VnFSpec defined in nfvo client module
# Output: API response
def deploy_vnf(vnf_spec):

    ni_nfvo_api = get_nfvo_vnf_api()
    api_response = ni_nfvo_api.deploy_vnf(vnf_spec)

    return api_response


# destroy_vnf(id): destory VNF instance in OpenStack environment
# Inpurt: ID of VNF instance
# Output: API response
def destroy_vnf(id):

    ni_nfvo_api = get_nfvo_vnf_api()
    api_response = ni_nfvo_api.destroy_vnf(id)

    return api_response


# measure_response_time(): send http requests from a source to a destination
# Input: Null, instead, configure config.yaml file in advance
# Output: response time
def measure_response_time():

    cnd_path = os.path.dirname(os.path.realpath(__file__))
    command = "./test_http_e2e.sh %s %s %s %s %s"
    command = "cd " + cnd_path + "/testing-tools && " + command

    command = (command % (cfg["sla_monitoring"]["src"],
                          cfg["sla_monitoring"]["ssh_id"],
                          cfg["sla_monitoring"]["ssh_pw"],
                          cfg["sla_monitoring"]["num_requests"],
                          cfg["sla_monitoring"]["dst"]))
    command = command + " | grep 'Time per request' | head -1 | awk '{print $4}'"

    response = subprocess.check_output(command, shell=True).strip().decode("utf-8")

    return float(response)

    #if response == '':
    #    ni_nfvo_sfc_api = get_nfvo_sfc_api()
    #    sfc_update_spec = ni_nfvo_client.SfcUpdateSpec() # SfcUpdateSpec | Sfc Update info.
    #    sfc_info = get_sfc_by_name("dy-sfc")
    #    sfc_update_spec.sfcr_ids = sfc_info.sfcr_ids
    #    sfc_update_spec.vnf_instance_ids = initial_sfc_info
    #    ni_nfvo_sfc_api.update_sfc(sfc_info.id, sfc_update_spec)
    #    return -10000
    #return float(response)


# get_sfc_prefix(sfc_info): get sfc_prefix from sfc_info
# Input: SFC Info.
# Output: sfc_prefix
def get_sfc_prefix(sfc_info):
    prefix = sfc_info.sfc_name.split(cfg["instance"]["prefix_splitter"])[0]
    prefix = prefix + cfg["instance"]["prefix_splitter"]

    return prefix


# update_sfc(sfc_info): Update SFC, main function to do auto-scaling
# Input: updated sfc_info, which includes additional instances or removed instances
# Output: none
def update_sfc(sfc_info):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()
    sfc_update_spec = ni_nfvo_client.SfcUpdateSpec() # SfcUpdateSpec | Sfc Update info.

    sfc_update_spec.sfcr_ids = sfc_info.sfcr_ids
    sfc_update_spec.vnf_instance_ids = sfc_info.vnf_instance_ids

    ni_nfvo_sfc_api.update_sfc(sfc_info.id, sfc_update_spec)


# check_active_instance(id): Check an instance whether it's status is ACTIVE
# Input: instance id
# Output: True or False
def check_active_instance(id):
    api = get_monitoring_api()
    status = api.get_vnf_instance(id).status

    if status == "ACTIVE":
        return True
    else:
        return False


# threshold_scaling(scaler): doing auto-scaling based on threshold
# Input: scaler
# Output: none
def threshold_scaling(scaler):

    start_time = dt.datetime.now()

    # Target SFC exist
    if get_sfc_by_name(scaler.get_sfc_name()):
        while scaler.get_active_flag():
            response_time = measure_response_time()

            # Set sclaing_flag to show it is out or in or maintain
            if response_time > scaler.get_threshold_out():
                print("Scaling-out!")
                scaling_flag = 1
            elif response_time < scaler.get_threshold_in():
                print("Scaling-in!")
                scaling_flag = -1
            else:
                print("Maintain!")
                scaling_flag = 0

            # Scale-in or out
            if scaling_flag != 0:
                sfc_info = get_sfc_by_name(scaler.get_sfc_name())
                sfc_prefix = get_sfc_prefix(sfc_info)
                sfc_vnfs = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
                del sfc_vnfs[0] # Flow classifier instance deletion
                vnf_info = get_vnf_info(sfc_prefix, sfc_vnfs)

                tier_status = get_tier_status(vnf_info, sfc_info)
                tier_index = get_target_tier(tier_status, scaling_flag)

                if tier_index > -1:
                    target_vnf_type = sfc_vnfs[tier_index]
                    tier_vnf_ids = sfc_info.vnf_instance_ids[tier_index]
                    num_tier_instances = len(tier_vnf_ids)

                    # Scaling-out
                    if scaling_flag > 0:
                        # If possible to deploy new VNF instance
                        if num_tier_instances < cfg["instance"]["max_number"]:
                            vnf_spec = get_nfvo_vnf_spec()
                            vnf_spec.vnf_name = sfc_prefix + target_vnf_type + cfg["instance"]["prefix_splitter"] + str(dt.datetime.now())
                            vnf_spec.image_id = cfg["image"][target_vnf_type]
                            instance_id = deploy_vnf(vnf_spec)

                            # Wait 1 minute until creating VNF instnace
                            for i in range(0, 30):
                                time.sleep(2)

                                # Success to create VNF instance
                                if check_active_instance(instance_id):
                                    tier_vnf_ids.append(instance_id)
                                    sfc_info.vnf_instance_ids[tier_index] = tier_vnf_ids
                                    update_sfc(sfc_info)
                                    break

                    # Scaling-in
                    elif scaling_flag < 0:
                        # If possible to remove VNF instance
                        if num_tier_instances > cfg["instance"]["min_number"]:
                            index = random.randrange(0, num_tier_instances)

                            instance_id = tier_vnf_ids[index]
                            tier_vnf_ids.remove(instance_id)
                            sfc_info.vnf_instance_ids[tier_index] = tier_vnf_ids
                            update_sfc(sfc_info)
                            destroy_vnf(instance_id)

            current_time = dt.datetime.now()

            if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
                scaler.set_active_flag(False)

            time.sleep(scaler.get_interval())


    # Delete AutoScaler object
    if scaler in scaler_list:
        scaler_list.remove(scaler)
        pprint("[Expire] Threshold Scaling")
    else:
        pprint("[Exit] Thresold Scaling")


# dqn-threshold(scaler): doing auto-scaling based on dqn
# Input: scaler
# Output: none
def dqn_scaling(scaler):

    start_time = dt.datetime.now()
    sfc_info = get_sfc_by_name(scaler.get_sfc_name())

    # Target SFC exist
    if sfc_info:
        sfc_prefix = get_sfc_prefix(sfc_info)
        sfc_vnfs = get_sfcr_by_id(sfc_info.sfcr_ids[-1]).nf_chain
        del sfc_vnfs[0] # Flow classifier instance deletion

        node_info = get_node_info()
        num_nodes = len(node_info)

        q = Qnet(len(sfc_vnfs), len(node_info), num_neurons)
        q_target = Qnet(len(sfc_vnfs), len(node_info), num_neurons)
        q_target.load_state_dict(q.state_dict()) # Q를 Target Q로 복사 (state_dict는 Model의 Weight 정보를 Dictionary 형태로 담고 있음)

        optimizer = optim.Adam(q.parameters(), lr=learning_rate)
        n_epi = 0

        # If there is dataset, read it
        memory = ReplayBuffer(buffer_limit)

        if scaler.has_dataset:
            if os.path.isfile("data/"+scaler.scaling_name):
                memory.readFromFile("data/"+scaler.scaling_name)
            else:
                f = open("data/"+scaler.scaling_name, 'w')
                f.close()

        # Start scaling
        while scaler.get_active_flag():
            epsilon = max(0.01, epsilon - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%

            sfc_info = get_sfc_by_name(scaler.get_sfc_name())
            vnf_info = get_vnf_info(sfc_prefix, sfc_vnfs)
            tier_status = get_tier_status(vnf_info, sfc_info)

            # Get state and select action
            s = get_state(tier_status)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            done = False

            # Check whether it is out or in or maintain
            if a < num_nodes:
                print("Scaling-out!")
                scaling_flag = 1
            elif a > num_nodes:
                print("Scaling-in!")
                scaling_flag = -1
            else:
                print("Maintain!")
                scaling_flag = 0

            # Scaling in or out
            if scaling_flag != 0:
                tier_index = get_target_tier(tier_status, scaling_flag)

                if tier_index > -1:
                    target_node = node_info[a%num_nodes]
                    target_vnf_type = sfc_vnfs[tier_index]
                    tier_vnf_ids = sfc_info.vnf_instance_ids[tier_index]
                    num_tier_instances = len(tier_vnf_ids)

                    # Scaling-out
                    if scaling_flag > 0:
                        # If possible to deploy new VNF instance
                        if num_tier_instances < cfg["instance"]["max_number"]:
                            vnf_spec = get_nfvo_vnf_spec()
                            vnf_spec.vnf_name = sfc_prefix + target_vnf_type + cfg["instance"]["prefix_splitter"] + str(dt.datetime.now())
                            vnf_spec.image_id = cfg["image"][target_vnf_type]
                            vnf_spec.node_name = target_node.name

                            instance_id = deploy_vnf(vnf_spec)

                            # Wait 1 minute until creating VNF instnace
                            for i in range(0, 30):
                                time.sleep(2)

                                # Success to create VNF instance
                                if check_active_instance(instance_id):
                                    tier_vnf_ids.append(instance_id)
                                    sfc_info.vnf_instance_ids[tier_index] = tier_vnf_ids
                                    update_sfc(sfc_info)
                                    done = True
                                    break

                    # Scaling-in
                    elif scaling_flag < 0:
                        # If possible to remove VNF instance
                        if num_tier_instances > cfg["instance"]["min_number"]:
                            for i in range(0, 10):
                                vnf_candidates = [ vnf_instance for vnf_instance in vnf_info if vnf_instance.node_id == target_node.name and vnf_instance.id in tier_vnf_ids]
                                num_vnf_candidates = len(vnf_candidates)

                                # Possible to remove VNF instance from a target node
                                if num_vnf_candidates > 0:
                                    index = random.randrange(0, num_vnf_candidates)
                                    instance_id = tier_vnf_ids[index]
                                    tier_vnf_ids.remove(instance_id)
                                    sfc_info.vnf_instance_ids[tier_index] = tier_vnf_ids
                                    update_sfc(sfc_info)
                                    destroy_vnf(instance_id)
                                    done = True
                                    break
                                # Re-select an action
                                else:
                                    a = q.sample_action(torch.from_numpy(s).float(), epsilon)

                                    if a < num_nodes:
                                        a = a + num_nodes + 1
                                        target_node = node_info[a%num_nodes]

                    # Maintain
                    else:
                        done = True

            # Prepare calculating rewards
            sfc_info = get_sfc_by_name(scaler.get_sfc_name())
            vnf_info = get_vnf_info(sfc_prefix, sfc_vnfs)
            tier_status = get_tier_status(vnf_info, sfc_info)

            s_prime = get_state(tier_status)
            r = calculate_reward(vnf_info, node_info, sfc_info, scaler.get_slo())

            done_mask = 1.0 if done else 0.0
            transition = (s,a,r,s_prime,done_mask)
            memory.put(transition)

            # If has_dataset, save transition
            if scaler.has_dataset:
                if os.path.isfile("data/"+scaler.scaling_name):
                    memory.writeToFile("data/"+scaler.scaling_name, transition)
                else:
                    f = open("data/"+scaler.scaling_name, 'w')
                    f.close()
                    memory.writeToFile("data/"+scaler.scaling_name, transition)

            if memory.size() > required_mem_size:
                train(q, q_target, memory, optimizer, gamma, batch_size)

            if n_epi % print_interval==0 and n_epi != 0:
                print("Target network updated!")
                q_target.load_state_dict(q.state_dict())

            current_time = dt.datetime.now()

            if scaler.get_duration() > 0 and (current_time-start_time).seconds > scaler.get_duration():
                scaler.set_active_flag(False)

            n_epi = n_epi+1

            time.sleep(scaler.get_interval())


    # Delete AutoScaler object
    if scaler in scaler_list:
        scaler_list.remove(scaler)
        pprint("[Expire] DQN Scaling")
    else:
        pprint("[Exit] DQN Scaling")


# calculate_reward(vnf_info, node_info, sfc_info, slo): calcuate reward about action
# Input: vnf_info, node_info, sfc_info, slo (get data to calculate reward)
# Output: calculated reward
def calculate_reward(vnf_info, node_info, sfc_info, slo):
    alpha = 1.2 # weight1
    beta =  1 # weight2
    sla_score = 0 # Check sla violation
    dist_vnf = 1 # Distribution of VNF instances
    dist_node = 1 # Distribution of used nodes (Node_used / Node_total)

    # Preprocessing: get vnf IDs placed in sfc
    sfc_vnf_ids = []
    for vnf_ids in sfc_info.vnf_instance_ids:
        sfc_vnf_ids = sfc_vnf_ids + vnf_ids

    # SLA violation check for reward
    response_time = measure_response_time()

    sla_score = -response_time/slo

    # Distribution of VNF instances,  Distribution of Node
    vnf_total = len(sfc_vnf_ids)
    vnf_deployed = [ vnf.node_id for vnf in vnf_info if vnf.id in sfc_vnf_ids ]
    node_total = len(node_info)
    node_used = 0

    for node in node_info:
        vnf_in_node = vnf_deployed.count(node.name) # the number of VNF instance in the node

        # Exist at least one instance in the node
        if vnf_in_node > 0:
            node_used = node_used + 1
            dist_vnf = dist_vnf * (vnf_in_node / vnf_total)
    dist_node = node_used / node_total

    # Calculation reward
    reward = sla_score + (alpha * dist_vnf * math.exp(-beta*dist_node))
    return reward

'''Unused now'''
# set_flow_classifier(sfcr_name, sfc_ip_prefix, nf_chain, source_client): create flow classifier in the testbed
# Input: flowclassifier name, flowclassifier ip prefix, list[list[each vnf id]], flowclassifier VM ID
# Output: response
def set_flow_classifier(sfcr_name, src_ip_prefix, nf_chain, source_client):

    ni_nfvo_sfcr_api = get_nfvo_sfcr_api()

    sfcr_spec = ni_nfvo_client.SfcrSpec(name=sfcr_name,
                                 src_ip_prefix=src_ip_prefix,
                                 nf_chain=nf_chain,
                                 source_client=source_client)

    api_response = ni_nfvo_sfcr_api.add_sfcr(sfcr_spec)

    return api_response

'''Unused now'''
# set_sfc(sfcr_id, sfc_name, sfc_path, vnfi_list): create sfc in the testbed
# Input: flowclassifier name, sfc name, sfc path, vnfi_info
# Output: response
def set_sfc(sfcr_id, sfc_name, sfc_path, vnfi_info):

    ni_nfvo_sfc_api = get_nfvo_sfc_api()

    vnf_instance_ids= []

    for vnfi in vnfi_info:
        for vnf in sfc_path:
            if sfc_path.index(vnf) == 0:
                continue

            if vnfi.name == vnf:
                vnf_instance_ids.append([ vnfi.id ])

    sfc_spec = ni_nfvo_client.SfcSpec(sfc_name=sfc_name,
                                   sfcr_ids=[ sfcr_id ],
                                   vnf_instance_ids=vnf_instance_ids)

    api_response = ni_nfvo_sfc_api.set_sfc(sfc_spec)

    return api_response
