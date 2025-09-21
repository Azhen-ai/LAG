"""
基于JSBSim的飞行仿真系统
包含飞机(AircraftSimulator)和导弹(MissileSimulator)的六自由度动力学仿真实现
核心功能：
1. 基于JSBSim引擎的固定翼飞机动力学仿真
2. 比例导引法（Proportional Navigation Guidance）导弹制导
3. 多实体协同作战管理
"""

import os
import logging
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from typing import Literal, Union, List

import jsbsim  # JSBSim飞行动力学引擎核心库
from .catalog import Property, Catalog  # 仿真属性目录管理
from ..utils.utils import get_root_dir, LLA2NEU, NEU2LLA  # 地理坐标转换工具

# 定义队伍颜色枚举类型（红蓝对抗场景）
TeamColors = Literal["Red", "Blue", "Green", "Violet", "Orange"]

class BaseSimulator(ABC):
    """仿真器抽象基类
    功能：
    - 定义所有仿真实体的公共属性和方法
    - 管理地理坐标系/东北天坐标系的状态量
    - 提供基础的生命周期管理方法
    """
    
    def __init__(self, uid: str, color: TeamColors, dt: float):
        """仿真器构造函数
        参数：
        uid: 5位十六进制唯一标识符（例：A0100）
        color: 队伍颜色标识（红蓝对抗场景） 
        dt: 仿真步长（单位：秒）
        """
        self.__uid = uid       # 唯一标识符（protected成员）
        self.__color = color   # 队伍颜色（protected成员）
        self.__dt = dt         # 仿真时间步长（protected成员）
        self.model = ""        # 载具模型名称
        # 地理坐标系状态量（经度°, 纬度°, 海拔m）
        self._geodetic = np.zeros(3)  
        # 东北天坐标系位置（北m, 东m, 天m）
        self._position = np.zeros(3)  
        # 姿态角（滚转rad, 俯仰rad, 偏航rad）
        self._posture = np.zeros(3)   
        # 速度向量（北m/s, 东m/s, 天m/s）
        self._velocity = np.zeros(3)  
        logging.debug(f"{self.__class__.__name__}:{self.__uid} is created!")

    @property
    def uid(self) -> str:
        """获取唯一标识符"""
        return self.__uid

    @property
    def color(self) -> str:
        """获取队伍颜色"""
        return self.__color

    @property
    def dt(self) -> float:
        """获取仿真步长"""
        return self.__dt

    def get_geodetic(self):
        """获取地理坐标系状态量
        返回：(经度°, 纬度°, 海拔m)的numpy数组
        """
        return self._geodetic

    def get_position(self):
        """获取东北天坐标系位置
        返回：(北m, 东m, 天m)的numpy数组
        """
        return self._position

    def get_rpy(self):
        """获取姿态角（欧拉角）
        返回：(滚转rad, 俯仰rad, 偏航rad)的numpy数组
        """
        return self._posture

    def get_velocity(self):
        """获取速度向量
        返回：(北m/s, 东m/s, 天m/s)的numpy数组
        """
        return self._velocity

    def reload(self):
        """重置状态量到初始值"""
        self._geodetic = np.zeros(3)
        self._position = np.zeros(3)
        self._posture = np.zeros(3)
        self._velocity = np.zeros(3)

    @abstractmethod
    def run(self, ​**kwargs):
        """仿真步进方法（需子类实现）"""
        pass

    def log(self):
        """生成状态日志报文
        格式示例：A0100,T=120.5|60.3|5000|10.2|5.1|30.0,Name=F16,Color=Red
        """
        lon, lat, alt = self.get_geodetic()
        # 弧度转角度（可视化需要）
        roll, pitch, yaw = self.get_rpy() * 180 / np.pi  
        log_msg = f"{self.uid},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
        log_msg += f"Name={self.model.upper()},"  # 型号名称大写
        log_msg += f"Color={self.color}"
        return log_msg

    @abstractmethod
    def close(self):
        """资源释放方法（需子类实现）"""
        pass

    def __del__(self):
        """析构函数，记录对象销毁日志"""
        logging.debug(f"{self.__class__.__name__}:{self.uid} is deleted!")


class AircraftSimulator(BaseSimulator):
    """飞机仿真器（继承自BaseSimulator）
    核心功能：
    - 集成JSBSim引擎实现六自由度飞行动力学仿真
    - 管理武器系统（导弹发射与拦截）
    - 处理协同作战关系（友军/敌军）
    """
    
    # 状态枚举常量
    ALIVE = 0    # 正常运行状态
    CRASH = 1    # 坠毁状态（低高度/极端姿态/过载）
    SHOTDOWN = 2 # 被导弹击落状态

    def __init__(self,
                 uid: str = "A0100",
                 color: TeamColors = "Red",
                 model: str = 'f16',
                 init_state: dict = {},
                 origin: tuple = (120.0, 60.0, 0.0),
                 sim_freq: int = 60, ​**kwargs):
        """飞机仿真器构造函数
        参数：
        model: JSBSim飞机模型名称（对应data目录下的xml文件）
        init_state: 初始状态字典（覆盖默认属性）
        origin: 全局战场原点（经度°, 纬度°, 海拔m）
        sim_freq: JSBSim积分频率（Hz）
        """
        super().__init__(uid, color, 1 / sim_freq)
        self.model = model  # 载具模型名称（如f16）
        self.init_state = init_state  # 自定义初始状态
        # 战场原点坐标（用于东北天坐标系转换）
        self.lon0, self.lat0, self.alt0 = origin  
        self.bloods = 100  # 生命值（百分比）
        self.__status = AircraftSimulator.ALIVE  # 初始状态
        
        # 武器系统配置（通过关键字参数传入）
        for key, value in kwargs.items():
            if key == 'num_missiles':
                self.num_missiles = value  # 最大载弹量
                self.num_left_missiles = self.num_missiles  # 剩余导弹数
                
        # 协同关系管理
        self.partners = []  # 友军列表（AircraftSimulator实例）
        self.enemies = []   # 敌军列表（AircraftSimulator实例）
        
        # 导弹系统管理
        self.launch_missiles = []   # 已发射导弹（MissileSimulator实例）
        self.under_missiles = []    # 来袭导弹（MissileSimulator实例）
        
        self.reload()  # 初始化JSBSim实例

    @property
    def is_alive(self):
        """判断是否存活"""
        return self.__status == AircraftSimulator.ALIVE

    @property
    def is_crash(self):
        """判断是否坠毁"""
        return self.__status == AircraftSimulator.CRASH

    @property
    def is_shotdown(self):
        """判断是否被击落"""
        return self.__status == AircraftSimulator.SHOTDOWN

    def crash(self):
        """设置为坠毁状态"""
        self.__status = AircraftSimulator.CRASH

    def shotdown(self):
        """设置为被击落状态"""
        self.__status = AircraftSimulator.SHOTDOWN

    def reload(self, new_state: Union[dict, None] = None, new_origin: Union[tuple, None] = None):
        """重置仿真器状态
        功能：
        1. 重新加载JSBSim实例
        2. 应用新的初始状态
        3. 重置武器系统状态
        """
        super().reload()  # 调用基类重置方法

        # 重置临时状态
        self.bloods = 100
        self.__status = AircraftSimulator.ALIVE
        self.launch_missiles.clear()  # 清空已发射导弹
        self.under_missiles.clear()   # 清空来袭导弹
        self.num_left_missiles = self.num_missiles  # 重置剩余导弹数

        # 初始化JSBSim FDM实例（参考网页6 JSBSim API）
        self.jsbsim_exec = jsbsim.FGFDMExec(os.path.join(get_root_dir(), 'data'))
        self.jsbsim_exec.set_debug_level(0)  # 关闭调试输出
        self.jsbsim_exec.load_model(self.model)  # 加载飞机模型
        
        # 注册JSBSim属性到Catalog（用于统一管理）
        Catalog.add_jsbsim_props(self.jsbsim_exec.query_property_catalog(""))  
        self.jsbsim_exec.set_dt(self.dt)  # 设置积分步长
        self.clear_defalut_condition()   # 清除默认初始条件

        # 应用新的初始状态（如果提供）
        if new_state is not None:
            self.init_state = new_state
        if new_origin is not None:
            self.lon0, self.lat0, self.alt0 = new_origin
            
        # 设置自定义初始属性    
        for key, value in self.init_state.items():
            self.set_property_value(Catalog[key], value)
            
        # 运行初始条件设置（参考网页6 JSBSim初始化流程）    
        success = self.jsbsim_exec.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

        # 推进系统初始化（发动机启动）
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        for j in range(n):
            propulsion.get_engine(j).init_running()
        propulsion.get_steady_state()  # 达到稳态
        
        self._update_properties()  # 更新内部状态量

    def clear_defalut_condition(self):
        """清除JSBSim默认初始条件
        设置标准初始状态：
        - 经度120°, 纬度60°
        - 海拔20000英尺（约6096米）
        - 航向0°
        - 初始速度800英尺/秒（约244m/s）
        """
        default_condition = {
            Catalog.ic_long_gc_deg: 120.0,  # 初始经度
            Catalog.ic_lat_geod_deg: 60.0,  # 初始纬度
            Catalog.ic_h_sl_ft: 20000,      # 初始海拔（英尺）
            Catalog.ic_psi_true_deg: 0.0,   # 初始航向角
            Catalog.ic_u_fps: 800.0,        # 机体X轴速度（英尺/秒）
            Catalog.ic_v_fps: 0.0,          # 机体Y轴速度
            Catalog.ic_w_fps: 0.0,          # 机体Z轴速度
            Catalog.ic_p_rad_sec: 0.0,      # 滚转角速度
            Catalog.ic_q_rad_sec: 0.0,      # 俯仰角速度
            Catalog.ic_r_rad_sec: 0.0,      # 偏航角速度
            Catalog.ic_roc_fpm: 0.0,        # 初始爬升率
            Catalog.ic_terrain_elevation_ft: 0,  # 地形高程
        }
        # 应用默认条件
        for prop, value in default_condition.items():
            self.set_property_value(prop, value)

    def run(self):
        """执行单步仿真
        返回值：
        bool: JSBSim是否继续运行（False表示达到终止条件）
        """
        if self.is_alive:
            if self.bloods <= 0:
                self.shotdown()  # 生命值耗尽则标记为击落
            result = self.jsbsim_exec.run()  # 执行JSBSim单步仿真
            if not result:
                raise RuntimeError("JSBSim failed.")
            self._update_properties()  # 更新状态量
            return result
        else:
            return True  # 非存活状态直接返回True

    def close(self):
        """释放资源"""
        if self.jsbsim_exec:
            self.jsbsim_exec = None  # 释放JSBSim实例
        self.partners = []  # 清空友军列表
        self.enemies = []   # 清空敌军列表

    def _update_properties(self):
        """更新内部状态量
        从JSBSim获取最新数据并转换坐标系：
        1. 地理坐标 → 东北天坐标系
        2. 机体速度 → 地面坐标系
        """
        # 获取地理坐标（经度,纬度,海拔）
        self._geodetic[:] = self.get_property_values([
            Catalog.position_long_gc_deg,
            Catalog.position_lat_geod_deg,
            Catalog.position_h_sl_m
        ])
        # 转换到东北天坐标系（相对战场原点）
        self._position[:] = LLA2NEU(*self._geodetic, self.lon0, self.lat0, self.alt0)
        
        # 获取姿态角（滚转,俯仰,偏航）单位弧度
        self._posture[:] = self.get_property_values([
            Catalog.attitude_roll_rad,
            Catalog.attitude_pitch_rad,
            Catalog.attitude_heading_true_rad,
        ])
        
        # 获取速度向量（北,东,地速）
        self._velocity[:] = self.get_property_values([
            Catalog.velocities_v_north_mps,
            Catalog.velocities_v_east_mps,
            Catalog.velocities_v_down_mps,
        ])
        # 转换地速为天速（向上为正）
        self._velocity[2] = -self._velocity[2]

    def get_sim_time(self):
        """获取JSBSim仿真时间"""
        return self.jsbsim_exec.get_sim_time()

    def get_property_values(self, props):
        """批量获取属性值
        参数：
        props: Property对象列表
        返回：属性值列表
        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """批量设置属性值
        参数：
        props: Property对象列表
        values: 对应属性值列表
        """
        if not len(props) == len(values):
            raise ValueError("属性与值数量不匹配")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """获取单个属性值
        参数：
        prop: Property对象
        返回：float型属性值
        """
        if isinstance(prop, Property):
            if prop.access == "R":  # 只读属性需要更新
                if prop.update:      # 如果有更新回调
                    prop.update(self)  
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        else:
            raise ValueError(f"无效属性类型: {type(prop)}")

    def set_property_value(self, prop, value):
        """设置属性值
        参数：
        prop: Property对象
        value: 要设置的数值（自动进行范围约束）
        """
        if isinstance(prop, Property):
            # 数值范围约束
            value = max(prop.min, min(prop.max, value))
            # 设置JSBSim属性
            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)
            # 处理可写属性的更新回调
            if "W" in prop.access and prop.update:
                prop.update(self)
        else:
            raise ValueError(f"无效属性类型: {type(prop)}")

    def check_missile_warning(self):
        """检查来袭导弹威胁
        返回：首个存活的来袭导弹实例，若无则返回None
        """
        for missile in self.under_missiles:
            if missile.is_alive:
                return missile
        return None


class MissileSimulator(BaseSimulator):
    """导弹仿真器（继承自BaseSimulator）
    核心功能：
    - 实现比例导引法（Proportional Navigation Guidance）
    - 六自由度动力学仿真
    - 毁伤效果判定与状态管理
    """
    
    # 状态枚举常量
    INACTIVE = -1  # 未激活状态
    LAUNCHED = 0   # 已发射状态
    HIT = 1        # 命中目标状态
    MISS = 2       # 脱靶状态

    @classmethod
    def create(cls, parent: AircraftSimulator, target: AircraftSimulator, 
              uid: str, missile_model: str = "AIM-9L"):
        """工厂方法创建导弹实例
        参数：
        parent: 发射载机实例
        target: 攻击目标实例
        uid: 导弹唯一标识
        missile_model: 导弹型号
        """
        assert parent.dt == target.dt, "时间步长不一致"
        missile = cls(uid, parent.color, missile_model, parent.dt)
        missile.launch(parent)  # 初始化导弹状态
        missile.target(target)  # 设置攻击目标
        return missile

    def __init__(self,
                 uid="A0101",
                 color="Red",
                 model="AIM-9L",
                 dt=1 / 12):
        """导弹仿真器构造函数
        参数：
        model: 导弹型号（对应气动参数）
        dt: 仿真步长（默认1/12秒）
        """
        super().__init__(uid, color, dt)
        self.__status = MissileSimulator.INACTIVE  # 初始状态
        self.model = model  # 导弹型号
        self.parent_aircraft = None  # 发射载机实例
        self.target_aircraft = None  # 攻击目标实例
        self.render_explosion = False  # 爆炸特效渲染标记

        # 导弹动力学参数（AIM-9L参数）
        self._g = 9.81       # 重力加速度(m/s²)
        self._t_max = 60     # 最大飞行时间(s)
        self._t_thrust = 3   # 发动机工作时间(s)
        self._Isp = 120      # 比冲(s)
        self._Length = 2.87  # 弹体长度(m)
        self._Diameter = 0.127  # 弹体直径(m)
        self._cD = 0.4       # 气动阻力系数
        self._m0 = 84        # 初始质量(kg)
        self._dm = 6         # 质量消耗率(kg/s)
        self._K = 3          # 比例导引系数
        self._nyz_max = 30   # 最大法向过载(g)
        self._Rc = 300       # 爆炸半径(m)
        self._v_min = 150    # 最小维持速度(m/s)

    @property
    def is_alive(self):
        """判断导弹是否在飞行中"""
        return self.__status == MissileSimulator.LAUNCHED

    @property
    def is_success(self):
        """判断是否命中目标"""
        return self.__status == MissileSimulator.HIT

    @property
    def is_done(self):
        """判断是否结束（命中或脱靶）"""
        return self.__status in (MissileSimulator.HIT, MissileSimulator.MISS)

    @property
    def Isp(self):
        """有效比冲（发动机工作期间有效）"""
        return self._Isp if self._t < self._t_thrust else 0

    @property
    def K(self):
        """动态比例导引系数（随时间衰减）"""
        return max(self._K * (self._t_max - self._t) / self._t_max, 0)

    @property
    def S(self):
        """等效横截面积（考虑攻角影响）"""
        S0 = np.pi * (self._Diameter / 2)​**2  # 基础横截面积
        # 考虑攻角影响的附加面积
        S0 += np.linalg.norm([np.sin(self._dtheta), np.sin(self._dphi)]) * self._Diameter * self._Length
        return S0

    @property
    def rho(self):
        """大气密度模型（指数近似）"""
        return 1.225 * np.exp(-self._geodetic[-1] / 9300)
        # 精确模型（分层大气）：
        # 参考https://www.cnblogs.com/pathjh/p/9127352.html

    @property
    def target_distance(self) -> float:
        """计算与目标的欧氏距离"""
        return np.linalg.norm(self.target_aircraft.get_position() - self.get_position())

    def launch(self, parent: AircraftSimulator):
        """导弹发射初始化
        参数：
        parent: 发射载机实例
        """
        self.parent_aircraft = parent
        self.parent_aircraft.launch_missiles.append(self)  # 注册到载机
        
        # 继承载机初始状态
        self._geodetic[:] = parent.get_geodetic()  # 地理坐标
        self._position[:] = parent.get_position()  # 东北天位置
        self._velocity[:] = parent.get_velocity()  # 速度向量
        self._posture[:] = parent.get_rpy()       # 姿态角
        self._posture[0] = 0  # 滚转角归零（导弹无滚转控制）
        
        # 设置战场原点与载机一致
        self.lon0, self.lat0, self.alt0 = parent.lon0, parent.lat0, parent.alt0
        
        # 初始化动力学参数
        self._t = 0        # 已飞行时间
        self._m = self._m0  # 当前质量
        self._dtheta, self._dphi = 0, 0  # 俯仰/偏航角速度
        self.__status = MissileSimulator.LAUNCHED  # 状态更新
        
        # 脱靶检测相关参数
        self._distance_pre = np.inf  # 上一时刻目标距离
        self._distance_increment = deque(maxlen=int(5 / self.dt))  # 5秒距离增量窗口
        self._left_t = int(1 / self.dt)  # 爆炸后残留时间

    def target(self, target: AircraftSimulator):
        """设置攻击目标
        参数：
        target: 目标飞机实例
        """
        self.target_aircraft = target  
        self.target_aircraft.under_missiles.append(self)  # 注册到目标

    def run(self):
        """执行导弹仿真步进"""
        self._t += self.dt  # 累计飞行时间
        action, distance = self._guidance()  # 比例导引计算过载指令
        
        # 脱靶条件检测
        self._distance_increment.append(distance > self._distance_pre)
        self._distance_pre = distance
        
        # 命中判定（距离小于爆炸半径且目标存活）
        if distance < self._Rc and self.target_aircraft.is_alive:
            self.__status = MissileSimulator.HIT
            self.target_aircraft.shotdown()  # 标记目标被击落
        
        # 脱靶条件（超时/速度过低/持续远离/目标已死）
        elif (self._t > self._t_max) or (np.linalg.norm(self.get_velocity()) < self._v_min) \
                or np.sum(self._distance_increment) >= self._distance_increment.maxlen \
                or not self.target_aircraft.is_alive:
            self.__status = MissileSimulator.MISS
        
        else:  # 正常飞行状态
            self._state_trans(action)  # 执行动力学更新

    def log(self):
        """生成日志报文（含爆炸特效指令）"""
        if self.is_alive:
            return super().log()  # 正常状态日志
        elif self.is_done and (not self.render_explosion):
            self.render_explosion = True  # 触发爆炸渲染
            # 生成爆炸特效指令
            lon, lat, alt = self.get_geodetic()
            roll, pitch, yaw = self.get_rpy() * 180 / np.pi
            log_msg = f"-{self.uid}\n"  # 移除导弹模型
            log_msg += f"{self.uid}F,T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Type=Misc+Explosion,Color={self.color},Radius={self._Rc}"
            return log_msg
        else:
            return None  # 已渲染过爆炸则不再输出

    def close(self):
        """释放目标关联"""
        self.target_aircraft = None

    def _guidance(self):
        """比例导引法实现（网页1/2/3/4原理）
        返回：
        tuple: (法向过载指令ny, nz), 当前目标距离
        """
        # 导弹状态参数
        x_m, y_m, z_m = self.get_position()
        dx_m, dy_m, dz_m = self.get_velocity()
        v_m = np.linalg.norm([dx_m, dy_m, dz_m])  # 导弹速度标量
        theta_m = np.arcsin(dz_m / v_m)  # 俯仰角
        
        # 目标状态参数
        x_t, y_t, z_t = self.target_aircraft.get_position()
        dx_t, dy_t, dz_t = self.target_aircraft.get_velocity()
        
        # 相对运动参数计算
        Rxy = np.linalg.norm([x_m - x_t, y_m - y_t])  # 水平面投影距离
        Rxyz = np.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # 三维距离
        
        # 视线角速度计算（网页2公式推导）
        dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
        deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
            (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
        
        # 法向过载指令生成（网页4核心公式）
        ny = self.K * v_m / self._g * np.cos(theta_m) * dbeta
        nz = self.K * v_m / self._g * deps + np.cos(theta_m)
        
        return np.clip([ny, nz], -self._nyz_max, self._nyz_max), Rxyz

    def _state_trans(self, action):
        """六自由度动力学方程积分（网页1/3原理）
        参数：
        action: 法向过载指令(ny, nz)
        """
        # 位置更新（欧拉积分）
        self._position[:] += self.dt * self.get_velocity()
        self._geodetic[:] = NEU2LLA(*self.get_position(), self.lon0, self.lat0, self.alt0)
        
        # 速度与姿态更新
        v = np.linalg.norm(self.get_velocity())  # 当前速度标量
        theta, phi = self.get_rpy()[1:]  # 当前俯仰/偏航角
        
        # 推力和阻力计算
        T = self._g * self.Isp * self._dm  # 发动机推力
        D = 0.5 * self._cD * self.S * self.rho * v**2  # 气动阻力
        
        # 轴向过载计算（网页3动力学模型）
        nx = (T - D) / (self._m * self._g)
        ny, nz = action  # 法向过载指令
        
        # 运动方程（网页1/3微分方程）
        dv = self._g * (nx - np.sin(theta))  # 切向加速度
        self._dphi = self._g / v * (ny / np.cos(theta))  # 偏航角速度
        self._dtheta = self._g / v * (nz - np.cos(theta))  # 俯仰角速度
        
        # 积分更新
        v += self.dt * dv  # 速度标量更新
        phi += self.dt * self._dphi  # 偏航角更新
        theta += self.dt * self._dtheta  # 俯仰角更新
        
        # 更新速度向量（球坐标系转笛卡尔）
        self._velocity[:] = np.array([
            v * np.cos(theta) * np.cos(phi),
            v * np.cos(theta) * np.sin(phi),
            v * np.sin(theta)
        ])
        # 更新姿态角（滚转保持为0）
        self._posture[:] = np.array([0, theta, phi])
        
        # 质量更新（发动机工作期间）
        if self._t < self._t_thrust:
            self._m = self._m - self.dt * self._dm