import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import random

import pandas as pd


def get_optimal(job_dict, opt_sign):
    if opt_sign == "max":
        return max(job_dict.values())
    elif opt_sign == "min":
        return min(job_dict.values())
    elif opt_sign == "random":
        if len(job_dict) <= 1:
            return min(job_dict.values())
        ran = np.random.randint(0, len(job_dict))
        i = 0
        for k, v in job_dict.items():
            if i == ran:
                return v
            i += 1


class JobEnv:
    def __init__(self, case_name, path, only_min=False):
        self.PDRs = {"SPT": "min", "MWKR": "max", "FDD/MWKR": "min", "MOPNR": "max", "LRM": "max", "FIFO": "max",
                     "LPT": "max", "LWKR": "min", "FDD/LWKR": "max", "LOPNR": "min", "SRM": "min", "LIFO": "min"}
        self.pdr_label = ["SPT", "MWKR", "FDD/MWKR", "MOPNR", "LRM", "FIFO",
                          "LPT", "LWKR", "FDD/LWKR", "LOPNR", "SRM", "LIFO"]
        self.machine_PDR = ["min", "max", "random"]
        self.case_name = case_name
        self.job_input = {}
        self.orders_of_job = {}
        file = path + case_name + ".fjs"
        with open(file, 'r') as f:
            user_line = f.readline()
            user_line = str(user_line).replace('\n', ' ')
            user_line = str(user_line).replace('\t', ' ')
            data = user_line.split(' ')
            while data.__contains__(""):
                data.remove("")
            self.m_n = list(map(float, data))

            for job in range(int(self.m_n[0])):
                user_line = f.readline()
                user_line = str(user_line).replace('\n', ' ')
                user_line = str(user_line).replace('\t', ' ')
                data = user_line.split(' ')
                while data.__contains__(""):
                    data.remove("")
                line_data = list(map(int, data))

                num_of_orders = line_data[0]
                self.orders_of_job[job] = num_of_orders
                k = 1
                for i in range(num_of_orders):
                    num_of_machines = line_data[k]
                    machines = []
                    processing_time = []
                    for j in range(num_of_machines):
                        machines.append(line_data[j*2+k+1])
                        processing_time.append(line_data[j*2+k+2])
                    self.job_input[job, i*2] = machines
                    self.job_input[job, i*2+1] = processing_time
                    k = k + 2 * num_of_machines + 1

        self.job_num = int(self.m_n[0])
        self.machine_num = int(self.m_n[1])
        self.only_min = only_min

        # self.action_num = len(self.pdr_label)
        self.action_num = int(len(self.pdr_label) / 2)
        self.max_job = self.job_num
        self.max_machine = self.machine_num
        self.current_time = 0  # current time
        self.scale = sum(self.orders_of_job.values())

        self.finished_jobs = None
        self.next_time_on_machine = None
        self.job_on_machine = None
        self.current_op_of_job = None
        self.assignable_job = None
        self.busy_job = None
        # state features are 2 variables
        self.state_num = self.max_job * 2
        self.state = None
        self.job_dict = None

        self.max_op_len = 0
        # find maximum operation length of all jobs
        for j in range(self.job_num):
            machines = self.orders_of_job[j]
            for i in range(machines):
                ops = self.job_input[j, i*2+1]
                if self.max_op_len < max(ops):
                    self.max_op_len = max(ops)

        self.last_release_time = None
        self.done = False
        self.reward = 0
        self.op_cnt = 0
        self.result_dict = {}

    def reset(self):
        self.current_time = 0  # current time
        self.next_time_on_machine = np.repeat(0, self.machine_num)
        self.job_on_machine = np.repeat(-1, self.machine_num)  # -1 implies idle machine
        self.current_op_of_job = np.repeat(0, self.job_num)  # current operation state of job
        self.assignable_job = np.ones(self.job_num, dtype=bool)  # whether a job is assignable
        self.busy_job = np.zeros(self.job_num, dtype=bool)  # whether a job is on process
        self.finished_jobs = np.zeros(self.job_num, dtype=bool)

        self.last_release_time = np.repeat(0, self.job_num)
        self.state = np.zeros(self.state_num, dtype=float)
        self.done = False
        self.op_cnt = 0
        self.job_dict = copy.deepcopy(self.job_input)
        return self._get_state()

    def find_process_time(self, job_id, current_op, sign):
        machine_set = self.job_dict[job_id, current_op * 2]
        process_time_set = self.job_dict[job_id, current_op * 2 + 1]
        valid_process_time = {}
        for i in range(len(machine_set)):
            if machine_set[i] > 0:
                valid_process_time[i] = process_time_set[i]
        if len(valid_process_time) <= 0:
            return 0
        return get_optimal(valid_process_time, sign)

    def get_feature(self, job_id, feature, sign):
        if feature == self.pdr_label[0] or feature == self.pdr_label[6]:
            return self.find_process_time(job_id, self.current_op_of_job[job_id], sign)
        elif feature == self.pdr_label[1] or feature == self.pdr_label[7]:
            work_remain = 0
            for i in range(self.orders_of_job[job_id] - self.current_op_of_job[job_id]):
                work_remain += self.find_process_time(job_id, i + self.current_op_of_job[job_id], sign)
            return work_remain
        elif feature == self.pdr_label[2] or feature == self.pdr_label[8]:
            work_remain = 0
            work_done = 0
            for i in range(self.orders_of_job[job_id] - self.current_op_of_job[job_id]):
                work_remain += self.find_process_time(job_id, i + self.current_op_of_job[job_id], sign)
            for k in range(self.current_op_of_job[job_id]):
                work_done += self.find_process_time(job_id, k, sign)
            if work_remain == 0:
                return 10000
            return work_done/work_remain
        elif feature == self.pdr_label[3] or feature == self.pdr_label[9]:
            return self.orders_of_job[job_id] - self.current_op_of_job[job_id]
        elif feature == self.pdr_label[4] or feature == self.pdr_label[10]:
            work_remain = 0
            for i in range(self.orders_of_job[job_id] - self.current_op_of_job[job_id] - 1):
                work_remain += self.find_process_time(job_id, i + self.current_op_of_job[job_id]+1, sign)
            return work_remain
        elif feature == self.pdr_label[5] or feature == self.pdr_label[11]:
            return self.current_time - self.last_release_time[job_id]
        return 0

    def _get_state(self):
        self.state[0:self.job_num] = self.current_op_of_job / self.machine_num
        self.state[self.max_job:self.state_num] = self.assignable_job
        return self.state.flatten()

    def get_selection(self, action):
        # action contains the PDRs for jobs and machines
        job_PDR = action
        PDR = [self.pdr_label[job_PDR], self.PDRs.get(self.pdr_label[job_PDR])]
        machine_selection = PDR[1]
        if self.only_min:
            machine_selection = self.machine_PDR[0]

        # allocate jobs according to PDRs
        job_dict = {}
        for i in range(self.job_num):
            if self.assignable_job[i]:
                job_dict[i] = self.get_feature(i, PDR[0], PDR[1])
        if len(job_dict) > 0:
            for key in job_dict.keys():
                if job_dict.get(key) == get_optimal(job_dict, PDR[1]):
                    return key, machine_selection

    def step(self, action):
        selected_job, selected_machine_PDR = self.get_selection(action)
        self.done = False
        self.reward = 0
        # action is operation
        self.allocate_job(selected_job, selected_machine_PDR)
        if self.stop():
            self.done = True
        return self._get_state(), self.reward/self.max_op_len, self.done

    def find_process_pairs(self, job_id, current_op, sign):
        # find the optimal machine according to job id and related PDR
        machine_set = self.job_dict[job_id, current_op * 2]
        process_time_set = self.job_dict[job_id, current_op * 2 + 1]
        for i in range(len(process_time_set)):
            if machine_set[i] > 0 and process_time_set[i] == self.find_process_time(job_id, current_op, sign):
                return machine_set[i], process_time_set[i]

    def allocate_job(self, job_id, sign):
        machine_id, process_time = self.find_process_pairs(job_id, self.current_op_of_job[job_id], sign)
        self.modify_machine(job_id, self.current_op_of_job[job_id], machine_id)
        # Attention that the job index starts from 0, while the machine index starts from 1
        self.job_on_machine[machine_id-1] = job_id
        self.op_cnt += 1

        start_time = self.next_time_on_machine[machine_id-1]
        self.next_time_on_machine[machine_id-1] += process_time
        end_time = start_time + process_time
        self.result_dict[job_id+1, machine_id] = start_time, end_time, process_time

        self.last_release_time[job_id] = self.current_time
        self.busy_job[job_id] = True
        self.assignable_job[job_id] = False

        # update the machine pool of all jobs
        for x in range(self.job_num):
            if not self.busy_job[x]:
                self.modify_machine(x, self.current_op_of_job[x], machine_id)
                if not self.assignable(x, self.current_op_of_job[x]):
                    self.assignable_job[x] = False
        # there is no assignable jobs after assigned a job and time advance is needed
        self.reward -= process_time
        while sum(self.assignable_job) == 0 and not self.stop():
            self.reward -= self.time_advance()
            self.release_machine()

    def time_advance(self):
        hole_len = 0
        min_next_time = min(self.next_time_on_machine)
        if self.current_time < min_next_time:
            self.current_time = min_next_time
        else:
            self.current_time = self.find_second_min()
        for machine in range(self.machine_num):
            dist_need_to_advance = self.current_time - self.next_time_on_machine[machine]
            if dist_need_to_advance > 0:
                self.next_time_on_machine[machine] += dist_need_to_advance
                hole_len += dist_need_to_advance
        return hole_len

    def release_machine(self):
        for k in range(self.machine_num):
            cur_job_id = self.job_on_machine[k]
            if cur_job_id >= 0 and self.current_time >= self.next_time_on_machine[k]:
                self.job_on_machine[k] = -1
                self.last_release_time[cur_job_id] = self.current_time
                for x in range(self.job_num):  # release jobs on this machine
                    if not self.busy_job[x] and not self.finished_jobs[x]:
                        self.modify_machine(x, self.current_op_of_job[x], -(k + 1))
                        if self.assignable(x, self.current_op_of_job[x]):
                            self.assignable_job[x] = True
                self.current_op_of_job[cur_job_id] += 1
                self.busy_job[cur_job_id] = False
                if self.current_op_of_job[cur_job_id] >= self.orders_of_job[cur_job_id]:
                    self.finished_jobs[cur_job_id] = True
                    self.busy_job[cur_job_id] = True
                    self.assignable_job[cur_job_id] = False
                else:
                    # if the next order of just released job has no machines to assign
                    for w in range(self.machine_num):
                        if self.job_on_machine[w] >= 0:
                            self.modify_machine(cur_job_id, self.current_op_of_job[cur_job_id], (w + 1))
                    if not self.assignable(cur_job_id, self.current_op_of_job[cur_job_id]):
                        self.assignable_job[cur_job_id] = False
                    else:
                        self.assignable_job[cur_job_id] = True

    def stop(self):
        if sum(self.current_op_of_job) < sum(self.orders_of_job.values()):
            return False
        return True

    def assignable(self, job_id, current_op):
        machine_set = self.job_dict[job_id, current_op * 2]
        for i in range(len(machine_set)):
            if machine_set[i] > 0 and self.job_on_machine[machine_set[i]-1] < 0:
                return True
        return False

    def modify_machine(self, job_id, current_op, machine_id):
        # using the negative value to make the occupied machine unavailable
        machine_set = self.job_dict[job_id, current_op * 2]
        new_machine_set = []
        for i in range(len(machine_set)):
            if machine_set[i] == machine_id:
                new_machine_set.append(-machine_set[i])
            else:
                new_machine_set.append(machine_set[i])
        self.job_dict[job_id, current_op * 2] = new_machine_set

    def modify_process_time(self, job_id, current_op, machine_id, process_time):
        machine_set = self.job_dict[job_id, current_op * 2]
        process_time_set = self.job_dict[job_id, current_op * 2 + 1]
        new_process_time_set = []
        for i in range(len(machine_set)):
            if machine_set[i] == machine_id:
                new_process_time_set.append(process_time)
            else:
                new_process_time_set.append(process_time_set[i])
        self.job_dict[job_id, current_op * 2 + 1] = new_process_time_set

    def find_second_min(self):
        min_time = min(self.next_time_on_machine)
        second_min_value = 100000
        for value in self.next_time_on_machine:
            if min_time < value < second_min_value:
                second_min_value = value
        if second_min_value == 100000:
            return min_time
        return second_min_value

    def draw_gantt(self):
        font_dict = {
            "style": "oblique",
            "weight": "bold",
            "color": "white",
            "size": 14
        }
        machine_labels = [" "]  # 生成y轴标签
        for i in range(self.machine_num):
            machine_labels.append("机器 " + str(i + 1))
        plt.figure(1)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
        colors = ['#%06X' % random.randint(0, 256 ** 3 - 1) for _ in range(30)]
        # print(len(self.result_dict))
        for k, v in self.result_dict.items():
            plt.barh(y=k[1]-1, width=v[2], left=v[0], edgecolor="black", color=colors[round(k[0])])
            plt.text(((v[0] + v[1]) / 2), k[1]-1, str(round(k[0])), fontdict=font_dict)
        plt.yticks([i - 1 for i in range(self.machine_num + 1)], machine_labels)
        plt.title(self.case_name)
        plt.xlabel("time")
        plt.ylabel("machines")
        plt.show()


if __name__ == '__main__':
    dir_path = "../MK/"
    PDR_label = ["SPT", "MWKR", "FDD/MWKR", "MOPNR", "LRM", "FIFO", "LPT", "LWKR", "FDD/LWKR", "LOPNR", "SRM", "LIFO"]
    results = pd.DataFrame(columns=PDR_label, dtype=int)
    for file_name in os.listdir(dir_path):
        title = file_name.split('.')[0]  # file name
        env = JobEnv(title, dir_path, only_min=False)
        case_result = []
        for pdr in range(len(PDR_label)):
            env.reset()
            cnt = 0
            while not env.stop():
                cnt += 1
                env.step(pdr)
            case_result.append(str(env.current_time))
            # env.draw_gantt()
        results.loc[title] = case_result
        print(title + str(case_result))
    results.to_csv("12PDR-MK.csv")
