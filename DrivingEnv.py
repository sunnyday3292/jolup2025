# ------------------------- DrivingEnv -------------------------
class DrivingEnv(gym.Env):
    def __init__(self, frames, yolo_model, lane_detector):
        super().__init__()
        if len(frames)==0: raise ValueError("frames empty")
        self.frames = frames
        self.current_idx = 0
        self.env_h, self.env_w = frames[0].shape[:2]
        self.car_x = self.env_w // 2
        self.yolo_model = yolo_model
        self.lane_detector = lane_detector
        
        self.action_space = spaces.Discrete(3)  # left, straight, right
        self.observation_space = spaces.Box(low=0, high=max(self.env_w, self.env_h),
                                            shape=(4,), dtype=np.float32)

    def apply_action(self, action):
        if action == 0:
            self.car_x -= 5
        elif action == 2:
            self.car_x += 5
        self.car_x = np.clip(self.car_x, 0, self.env_w)

    def reset(self):
        self.current_idx = 0
        self.car_x = self.env_w // 2
        return np.array([self.car_x,0,0,0], dtype=np.float32)

    def get_state(self):
        frame = self.frames[self.current_idx]
        # 장애물 탐지
        self.detected_obstacles = detect_obstacles(frame)
        # 차선 감지
        lanes = self.lane_detector.detect_lanes(frame)
        self.lane_detected = lanes is not None
        if self.lane_detected:
            self.left_lane_x, self.right_lane_x = lanes[0], lanes[1]
        else:
            self.left_lane_x = 0
            self.right_lane_x = self.env_w
        # nearest obstacle distance
        nearest_dist = min([self.env_h - y for x,y in self.detected_obstacles], default=self.env_h)
        return np.array([self.car_x, self.left_lane_x, self.right_lane_x, nearest_dist], dtype=np.float32)

    def step(self, action):
        self.apply_action(action)
        state = self.get_state()

        # Reward 계산
        reward = 0
        if self.lane_detected:
            reward += 1  # 차선 따라가기 보상
        
        for x, y in self.detected_obstacles:
            if self.left_lane_x < x < self.right_lane_x and y > self.env_h * 0.5:
                reward -= 5  # 차선 안 장애물 패널티
        
        done = False
        # 충돌 처리 (car_x가 차선 밖이면 충돌 가정)
        if self.car_x < self.left_lane_x or self.car_x > self.right_lane_x:
            reward -= 20
            done = True

        self.current_idx += 1
        if self.current_idx >= len(self.frames):
            done = True

        return state, reward, done, {}
