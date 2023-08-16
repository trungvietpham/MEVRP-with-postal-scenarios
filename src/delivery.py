import json
from time import time
import copy

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
from python_tsp.heuristics import solve_tsp_simulated_annealing

class Delivery:
    def __init__(self, delivery_nodes: dict[str, list], start_node: dict[str, list], vehicle_list: dict[str, list], distance_matrix, linkage, clustering_type):
        '''
        pickup_nodes: [lat, long, starttime, endtime, demand]
        vehicle_list: []
        '''
        self.delivery_nodes = delivery_nodes
        self.start_node = start_node
        self.vehicle_list = vehicle_list
        self.distance_matrix = np.array(distance_matrix)
        self.linkage = linkage
        self.clustering_type = clustering_type
        return
    
    def clustering(self, X, n_cluster: int, strategy='kmeans', linkage='ward'):
        if strategy == 'kmeans':
            model = KMeans(n_clusters=n_cluster, random_state=42, n_init=5)
            model.fit(X)
            output = model.predict(X)
        elif strategy == 'hierarchical':
            # model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
            model = AgglomerativeClustering(n_clusters=n_cluster, linkage=linkage)
            output = model.fit_predict(X)
        return output
    
    def tsp(self, distance_matrix, tpe=None):
        if tpe is None or tpe == 'bitmasking':
            return solve_tsp_dynamic_programming(distance_matrix)
        elif tpe == 'local_search':
            return solve_tsp_local_search(distance_matrix)
    
    def tsp_no_clustering(self, distance_matrix):
        return solve_tsp_local_search(distance_matrix)
    
    def tsp_no_clustering2(self, distance_matrix):
        return solve_tsp_simulated_annealing(distance_matrix)
    
    def to_array(self, a_dict: dict[str, list]):
        res = []
        self.converse_map = [] # map index với code
        self.code_map = {} # Map code với index
        for i, code in enumerate(a_dict): 
            res.append(a_dict[code])
            self.code_map[code] = i
            self.converse_map.append(code)
        return np.array(res)
    
    def find_nearest_node(self, current_node: str, candidate_nodes: list[str], all_node_info: dict[str, list]):
        current_node_vector = all_node_info[current_node]
        best = 1e9
        best_code = ''
        for code in candidate_nodes:
            # candidate_distance = np.linalg.norm(np.array(current_node_vector) - np.array(all_node_info[code]))
            candidate_distance = self.distance_matrix[int(self.code_map[current_node])][int(self.code_map[code])]
            if candidate_distance<best: 
                best = candidate_distance
                best_code = code
        return best_code, best
    
    def transform_node_vector(self, all_node_info):
        result = {}
        for code in all_node_info:
            result[code] = all_node_info[code][:2]
        self.all_node_location = result
        return result
    
    def find_best_fit_vehicle(self, remain_route: list[str], demand_by_route: list[float]):
        '''
        Tìm tập xe gần nhất so với điểm bắt đầu
        Return: id xe gần nhất, điểm kết thúc ứng với xe
        remain_route: chặng đường còn lại cần chạy
        demand_by_route: Khối lượng đơn hàng ứng với các điểm trong remain_route
        '''
        # print('Find best fit vehicle')
        # Tìm node gần với điểm hiện tại nhất, điểm hiện tại là remain_route[0]
        best, best_code = 1e9, ''
        current_node = remain_route[0]
        vehicle_manager_node_index = 1
        vehicle_status_index = 2
        
        for v_code, v_array in self.vehicle_list.items():
            if v_array[vehicle_status_index] == 0: continue
            if str(v_array[vehicle_manager_node_index]) not in self.all_node_location: continue
            if self.distance_matrix[int(self.code_map[str(current_node)])][int(self.code_map[str(v_array[vehicle_manager_node_index])])] < best: 
                best = self.distance_matrix[int(self.code_map[str(current_node)])][int(self.code_map[str(v_array[vehicle_manager_node_index])])]
                best_code = v_array[vehicle_manager_node_index]
        
        if best == 1e9: 
            # Nếu không còn xe thì đặt trạng thái của tất cả các xe về available, chạy lại find_best_fit_vehicle
            for v_code, v_array in self.vehicle_list.items():
                v_array[vehicle_status_index] = 1
            return self.find_best_fit_vehicle(remain_route, demand_by_route)
        
        '''Tìm những xe nằm gần điểm hiện tại nhất, 
        những điểm nằm gần là những điểm có khoảng cách < `best` + `epsilon` 
        '''   
        epsilon = 20
        n_try = 0
        while True: 
            res = []
            capacity = []
            print(f"Best: {best}")
            for v_code, v_array in self.vehicle_list.items():
                if v_array[vehicle_status_index] == 0: continue
                if v_array[vehicle_manager_node_index] not in self.all_node_location: continue
                if self.distance_matrix[int(self.code_map[current_node])][int(self.code_map[str(v_array[vehicle_manager_node_index])])] < best + epsilon:
                    res.append(v_code)
                    capacity.append(v_array[0])
            
            if len(capacity) == 0: 
                epsilon+=20
                n_try += 1
                continue
            # Sắp xếp lại mảng theo chiều tăng dần capacity
            # print(f"capacity: {capacity}, res: {res}")
            zipped_list = zip(capacity, res)
            sorted_pairs = sorted(zipped_list)
            tuples = zip(*sorted_pairs)
            capacity, res = [list(t) for t in tuples]
            print(f"Try: {n_try}, capacity: {capacity}, demand: {demand_by_route}")
            if self.vehicle_list[res[-1]][0] > demand_by_route[0]: break
            if n_try == 7: return res[-1], 1
            epsilon+=20
            n_try+=1
        
            
        # Trả về xe phù hợp nhất, update trạng thái cho xe thành 0 (not availavle)
        if capacity[-1] < np.sum(demand_by_route): # Xe có tải trọng lớn nhất vẫn nhỏ hơn demand của route
            self.vehicle_list[res[-1]][vehicle_status_index] = 0
            demand_for_check = 0
            for i, val in enumerate(demand_by_route): 
                demand_for_check += val
                if demand_for_check > capacity[-1]: break
            return res[-1], i-1
        
        else: 
            for i in range(len(capacity)):
                if capacity[len(capacity) - 1 - i] < np.sum(demand_by_route): break
            if i>len(capacity): i=len(capacity)
            print(f"res: {res}")
            self.vehicle_list[res[len(capacity)-i-1]][vehicle_status_index] = 0
            
            return  res[len(capacity)-i-1], len(demand_by_route) - 1
    
    def get_route_length(self, route: list[str], start_node: str):
        '''
        route: list các code của node 
        ''' 
        res = self.distance_matrix[self.code_map[start_node]][self.code_map[route[0]]]
        print(f"Start dis: {res}")
        # print(f" code map: {self.code_map.keys()}")
        for i in range(len(route)-1):
            # print(f"route i: {route[i]}, route i+1: {route[i+1]}")
            # print(f"code map: {self.code_map[str(route[i])]}, {self.code_map[str(route[i+1])]}")
            res += self.distance_matrix[int(self.code_map[str(route[i])])][int(self.code_map[str(route[i+1])])]
        print(f"Final res: {res}")
        return res
    
    def split_route(self, all_route, all_node):
        # Update demand_by_route
        route_list = []
        demand_by_routes = []
        for r in all_route:
            route_list += r
        for code in route_list:
            # print(code)
            if len(all_node[code]) >= 5:
                demand_by_routes.append(all_node[code][4])
            else: demand_by_routes.append(0)
        # Cắt nhỏ route, lựa chọn xe, tính quãng đường xe di chuyển
        '''
        Tìm tập các xe gần với điểm hiện tại nhất, 
            Nếu có xe có thể chở được hết route thì chọn xe có sức chứa nhỏ nhất vừa đủ
            Nếu không thì chọn xe có tải lớn nhất để chạy, cắt route tại điểm cuối cùng, xe chạy về GD1,
                Lặp lại từ bước tìm tập xe, cho tới khi hết route
        '''
        # print(route_list)
        # input('demand')
        distance_res = []
        percentage_res = []
        cost_res = []
        vehicle_route = {}
        remain_route = route_list.copy()
        remain_demand = demand_by_routes.copy()
        vehicle_list = copy.deepcopy(self.vehicle_list)
        fix_start_node = route_list[0]
        x = 0
        while True:
            print(f'X={x}') 
            if len(remain_route) == 0: break
            # print(remain_route)
            # Tìm điểm cận trên cho 1 route có thể (<200km)
            idx = -1
            for i in range(1, len(remain_route)):
                print(f"Remain route i: {remain_route[1:i+1]}")
                if self.get_route_length(remain_route[1:i+1], remain_route[1]) > 200:
                    idx = i
                    break
            if idx == 0: idx = 2
            elif idx == -1: idx = len(remain_route)
            print(f'idx: {idx}')
            vehicle_id, end_index = self.find_best_fit_vehicle(remain_route[:idx], remain_demand[:idx])

            percentage = []
            current_demand = 0
            # print(remain_demand)
            # input('remain')
            for j in range(end_index+1):
                current_demand+=remain_demand[j]
                percentage.append(current_demand/self.vehicle_list[vehicle_id][0])
            percentage_res.append(np.mean(percentage))
            # print(percentage)
            # input('percentage')

            child_routes = remain_route[: end_index+1]
            vehicle_route[vehicle_id] = [vehicle_list[vehicle_id][1]] + child_routes
            dis = self.get_route_length(child_routes, vehicle_list[vehicle_id][1])
            # dis += self.distance_matrix[int(self.code_map[child_routes[-1]])][int(self.code_map[list(self.return_node.keys())[0]])]
            distance_res.append(dis)
            cost_res.append(distance_res[-1] * vehicle_list[vehicle_id][0]/percentage_res[-1])
            if end_index >= len(remain_demand) - 1: break
            # if remain_demand[end_index] == demand_by_routes[-1]: break
            remain_route = [fix_start_node] + remain_route[end_index+1:]
            remain_demand = [0] + remain_demand[end_index+1:]
            x+=1
        for v in self.vehicle_list:
            self.vehicle_list[v][2] = 1
        return np.array(distance_res), np.array(percentage_res), np.array(cost_res), vehicle_route

    def execute2(self, province_code, write_type):
        # Các biến trả về
        distance_res = []
        routes_res = []
        time_res = []
        n_clusters = int(len(self.delivery_nodes) // 15 + 1)
        all_node = self.start_node.copy()
        all_node.update(self.delivery_nodes)
        X = self.to_array(all_node)
        # print(f"Length X: {len(X)}, length code_map: {len(self.code_map)}")
        X = X[:,:2]
        scaler = preprocessing.MinMaxScaler()
        X_normalized = scaler.fit_transform(X)
        time1 = time()
        output = np.array(self.clustering(X_normalized, n_clusters, strategy=self.clustering_type, linkage=self.linkage))
        # print(f"\tClustering time: {time() - time1}, number of cluster: {n_clusters}")
        # print(f"Length output: {len(output)}, length code_map: {len(self.code_map)}")
        # print(output)
        reverse = {}
        for i in range(n_clusters):
            reverse[i] = []
        
        for i, o in enumerate(output):
            reverse[int(o)].append(i)
        current_node = list(self.start_node.keys())[0]
        candidate_nodes = list(self.delivery_nodes.keys()) + list(self.start_node.keys())
        X_location = self.transform_node_vector(all_node)
        current_all_node = X_location.copy()
        
        for index in range(n_clusters):
            nearest_node, dis = self.find_nearest_node(current_node, candidate_nodes, current_all_node)
            distance_res.append(dis)
            
            i = int(output[int(self.code_map[nearest_node])]) # Lấy cluster label của nearest_node làm index
            # Đổi nearest_node thành node đầu tiên trong list reverse[i]
            if len(reverse[i]) > 1 and reverse[i].index(self.code_map[nearest_node]) != 0:
                tmp = reverse[i][0]
                reverse[i].remove(self.code_map[nearest_node])
                reverse[i][0] = self.code_map[nearest_node]
                reverse[i].append(tmp)
            
            # Tạo distance matrix
            i_distance_matrix = np.zeros((len(reverse[i]), len(reverse[i])))
            for j in range(len(reverse[i])):
                for k in range(len(reverse[i])):
                    i_distance_matrix[j][k] = self.distance_matrix[int(reverse[i][j])][int(reverse[i][k])]
            for j in range(len(reverse[i])):
                i_distance_matrix[j][0] = 0
            
            # TSP: 
            if len(reverse[i]) <= 17: tpe = 'bitmasking'
            else: tpe = 'local_search'
            time1 = time()
            routes, distance = self.tsp(i_distance_matrix, tpe)
            # print(f"\t{i}: {len(reverse[i])}")
            # print(f"\t\tTSP time: {time() - time1}")
            time_res.append(time() - time1)
            distance_res.append(distance)
            tmp = []
            for r in routes: 
                tmp.append(self.converse_map[int(reverse[i][int(r)])])
            routes_res.append(tmp)
            
            # Update các biến
            current_node = self.converse_map[int(reverse[i][int(routes[-1])])]
            for r in range(len(reverse[i])):
                # print(self.converse_map[int(reverse[i][int(r)])])
                try: candidate_nodes.remove(self.converse_map[int(reverse[i][r])])
                except: 
                    print(int(reverse[i][r]))
                    print(self.converse_map[int(reverse[i][r])])
                    print(self.converse_map[int(reverse[i][r])] in candidate_nodes)
                    raise Exception()
            # print('-'*100)
        
        distance_res, percentage_res, cost_res, vehicle_routes = self.split_route(routes_res, all_node)
        
        # Output kết quả ra file
        '''
        Output các thông tin: 
        Các cụm: số node trong cụm, tổng khoảng cách di chuyển trong cụm
        '''
        out_fname = f'scenarios/delivery_{self.clustering_type}_{self.linkage}.csv'
        dict_return = [province_code, len(self.delivery_nodes)+1, np.sum(distance_res), np.sum(time_res), np.mean(percentage_res), np.sum(cost_res)]
        with open(out_fname, write_type) as f: 
            f.write(f"{dict_return[0]},{dict_return[1]},{dict_return[2]},{dict_return[3]},{dict_return[4]}, {dict_return[5]}\n")
                
        # Tính TSP thuần ko kmeans để so sánh
        time1 = time()
        no_clustering_distance_matrix = self.distance_matrix.copy()
        for i in range(len(no_clustering_distance_matrix)):
            no_clustering_distance_matrix[i][0] = 0
        route_no_clustering, distance_no_clustering = self.tsp_no_clustering(copy.deepcopy(no_clustering_distance_matrix))
        route_list = [self.converse_map[int(i)] for i in route_no_clustering]
        distance_no_clustering, percentage_res, cost_res, vehicle_route_2 = self.split_route([route_list], all_node)
        with open('scenarios/delivery_no_clustering_local_search.csv', write_type) as f: 
            f.write(f"{dict_return[0]},{dict_return[1]},{np.sum(distance_no_clustering)},{time()-time1},{np.mean(percentage_res)},{np.sum(cost_res)}\n")
        
        # time1 = time()
        # route_no_clustering, distance_no_clustering = self.tsp_no_clustering2(copy.deepcopy(no_clustering_distance_matrix))
        # route_list = [self.converse_map[int(i)] for i in route_no_clustering]
        # distance_no_clustering, percentage_res = self.split_route([route_list], all_node)
        # with open('scenarios/delivery_no_clustering_simulate_annealing.csv', write_type) as f: 
        #     f.write(f"{dict_return[0]},{dict_return[1]},{np.sum(distance_no_clustering)},{time()-time1}, {np.mean(percentage_res)}\n")
        
        # Trả về kết quả (để update)
        return vehicle_routes, distance_res