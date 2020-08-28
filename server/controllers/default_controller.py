import connexion
import six

from server import util
from auto_scaling import *
from server.models.scaling_info import DQN_ScalingInfo
from server.models.scaling_info import Threshold_ScalingInfo
from server.models.scaling_info import AutoScaler

def get_all_scaling():

    response = []

    for process in scaler_list:
        response.append(process.get_info())

    return response

def get_scaling(name):
    response = [ process.get_info() for process in scaler_list if process.scaling_name == name]

    return response

def create_threshold_scaling(body):
    if connexion.request.is_json:
        body = Threshold_ScalingInfo.from_dict(connexion.request.get_json())
        response = AutoScaler(body, "threshold")
        scaler_list.append(response)

        threading.Thread(target=threshold_scaling, args=(response,)).start()

    return response.get_info()

def create_dqn_scaling(body):
    if connexion.request.is_json:
        body = DQN_ScalingInfo.from_dict(connexion.request.get_json())
        response = AutoScaler(body, "dqn")
        scaler_list.append(response)

        threading.Thread(target=dqn_scaling, args=(response,)).start()

    return response.get_info()

def delete_scaling(name):
    index = -1
    response = []

    for process in scaler_list:
        if process.scaling_name == name:
            index = scaler_list.index(process)
            break

    if index > -1:
        response = scaler_list[index].get_info()
        scaler_list[index].set_active_flag(False)
        scaler_list.remove(scaler_list[index])

    return response
