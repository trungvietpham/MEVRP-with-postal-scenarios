import copy
import json
import time 

import numpy as np

from Intermediate import Intermediate
from Visualize import Visualize
from pickup import Pickup
from transship import Transship
from delivery import Delivery


class Solution2:
    def __init__(self) -> None:
        return
    
    def get_to_go_distance(self, default_node_array: dict, to_go_node_array: dict, default_distance_matrix):
        '''
        Tạo distance_matrix giữa các node to_go_node
        default_node_array: Thông tin các node của 1 tỉnh
        to_go_node_array: Các node cần ghé thăm trong 1 tỉnh
        default_distance_matrix: ma trận khoảng cách tới tất cả các điểm trong 1 tỉnh
        
        '''
        white_list = []
        step = 0
        for i in range(len(default_distance_matrix) - len(default_node_array)):
            white_list.append(i)
        default_code = list(default_node_array.keys())
        to_go_code = list(to_go_node_array.keys())
        for i, code in enumerate(default_code):
            if code not in to_go_code:
                continue
            white_list.append(i)
        
        to_go_distance_matrix = np.zeros((len(white_list), len(white_list)))
        for i in range(len(white_list)): 
            for j in range(len(white_list)):
                to_go_distance_matrix[i][j] = default_distance_matrix[int(white_list[i])][int(white_list[j])]
        return to_go_distance_matrix

    def find_solution(self):
        with open('web/nevrpWeb/algo_log.txt') as f:
            data = f.readlines()
        clustering_type = data[0][:-1]
        linkage = data[1]
        print(f"{clustering_type}, {linkage}")
        # input()
        intermediate = Intermediate(r'data\vehicles.tsv', r'data\node.tsv', r'data\correlations.tsv', r'data\order.tsv')
        t1 = time.time()
        node, vehicle, order_array, correlation = intermediate.execute()
        print(f"Read data done after {time.time() - t1:0.1f} second")
        # input()
        # linkage = ['', 'ward', 'complete', 'average', 'single']
        # clustering_type = ['kmeans', 'hierarchical', 'hierarchical', 'hierarchical', 'hierarchical']
        

        

        
        # Chạy 1 lượt qua các giải thuật: Kmeans, hierarchical clustering với 4 linkage
        # for i in range(len(linkage)):
        # print(f"Algorithm: {clustering_type[i]}, linkage: {linkage[i]}")
        # Pickup phase: sender GD2 --> sender GD1
        write_type = 'w'
        node_demand = intermediate.get_node_demand('pickup', order_array) # Lấy thông tin node demand
        node_updated = intermediate.update_node_array('pickup', copy.deepcopy(node), node_demand) # Cập nhật node_demand vào thông tin các node 
        to_go_nodes = intermediate.get_to_go_node(node_updated)
        phase1_vehicle_routes = {}
        phase1_res = {}
        for province_code in to_go_nodes: 
            print(province_code)
            to_go_distance_matrix = self.get_to_go_distance(node_updated[province_code]['GD2'], to_go_nodes[province_code]['GD2'], correlation[province_code])
            phase1 = Pickup(copy.deepcopy(to_go_nodes[province_code]['GD2']), copy.deepcopy(node_updated[province_code]['GD1']), copy.deepcopy(vehicle[province_code]), to_go_distance_matrix, linkage, clustering_type)
            vehicle_route, distance_res = phase1.execute2(province_code, write_type)
            phase1_vehicle_routes.update(vehicle_route)
            idx = 0
            for key in vehicle_route: 
                phase1_res[key] = distance_res[idx]
                idx+=1
            # phase1_res.update(distance_res)
            write_type = 'a'
        print(f"\tDone pickup") 
        with open('scenarios/sender_side.json', 'w') as f:
            json.dump({"vehicle": phase1_vehicle_routes, 'distance': phase1_res}, f, indent=4)


        node_demand = intermediate.get_node_demand('transship', order_array) # Lấy thông tin node demand
        node_updated = intermediate.update_node_array('transship', copy.deepcopy(node), node_demand) # Cập nhật node_demand vào thông tin các node 
        all_gd1 = {}
        for province_code in node_updated: 
            all_gd1.update(node_updated[province_code]['GD1'])
        write_type = 'w'
        transship_phase = Transship(copy.deepcopy(all_gd1), vehicle[10], correlation['parent'], clustering_type=clustering_type, linkage=linkage)
        transship_phase.execute2(0, write_type)
        write_type = 'a'
        print(f"\tDone transship")

        node_demand = intermediate.get_node_demand('delivery', order_array) # Lấy thông tin node demand
        node_updated = intermediate.update_node_array('delivery', copy.deepcopy(node), node_demand) # Cập nhật node_demand vào thông tin các node 
        write_type = 'w'
        to_go_nodes = intermediate.get_to_go_node(node_updated)
        phase3_vehicle_routes = {}
        phase3_res = {}
        for province_code in to_go_nodes: 
            if 'GD2' not in to_go_nodes[province_code]: continue
            print(province_code)
            to_go_distance_matrix = self.get_to_go_distance(node_updated[province_code]['GD2'], to_go_nodes[province_code]['GD2'], correlation[province_code])
            phase3 = Delivery(copy.deepcopy(to_go_nodes[province_code]['GD2']), copy.deepcopy(node_updated[province_code]['GD1']), copy.deepcopy(vehicle[province_code]), to_go_distance_matrix, linkage, clustering_type)
            vehicle_route, distance_res = phase3.execute2(province_code, write_type)
            phase3_vehicle_routes.update(vehicle_route)
            idx = 0
            for key in vehicle_route: 
                phase3_res[key] = distance_res[idx]
                idx+=1
            # phase3_res.update(distance_res)
            write_type = 'a'
        print(f"\tDone delivery")
        with open('scenarios/receiver_side.json', 'w') as f:
            json.dump({"vehicle": phase3_vehicle_routes, "distance": phase3_res}, f, indent=4)
        
        visualize = Visualize()
        vehicle_path = visualize.get_vehicle_route()
        visualize.output_to_file(vehicle_path, '../scenarios/vehicle.json')
        



            