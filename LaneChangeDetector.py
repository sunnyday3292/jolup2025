class LaneChangeDetector:
    def __init__(self):
        self.prev_lane_state = "center"
        self.change_counter = 0
        self.threshold_frames = 5
        self.margin = 50
        self.vanish_history = []
        self.lane_width_history = []

        # 검출 상태
        self.left_detect = False
        self.right_detect = False
        self.left_m = self.right_m = 0
        self.left_b = self.right_b = (0,0)

        # 이전 프레임 lane 좌표 저장
        self.prev_lanes = [None, None]

    # 색상 필터링 (밝기/채도 낮은 것도 포함)
    def filter_colors(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0,0,180])
        upper_white = np.array([180,30,255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        lower_yellow = np.array([18,50,50])
        upper_yellow = np.array([30,255,255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        filtered = cv2.bitwise_and(image, image, mask=mask)
        return filtered

    # ROI
    def limit_region(self, image):
        height, width = image.shape[:2]
        mask = np.zeros_like(image)
        polygon = np.array([[
            (0,height),
            (width//2,int(height*0.55)),
            (width,height)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi = cv2.bitwise_and(image, mask)
        return roi

    # 허프 직선
    def houghLines(self, image):
        return cv2.HoughLinesP(image, 1, np.pi/180, 40, minLineLength=30, maxLineGap=120)

    # 좌우 차선 분리
    def separateLine(self, image, lines):
        left,right=[],[]
        height,width=image.shape[:2]
        self.img_center=width//2
        slope_thresh=0.3

        self.left_detect=self.right_detect=False

        for line in lines:
            x1,y1,x2,y2=line[0]
            slope=(y2-y1)/(x2-x1+1e-6)
            if abs(slope)<slope_thresh:
                continue
            if slope>0 and x1>self.img_center and x2>self.img_center:
                right.append(line)
                self.right_detect=True
            elif slope<0 and x1<self.img_center and x2<self.img_center:
                left.append(line)
                self.left_detect=True
        return left,right

    # 회귀선 계산
    def regression(self, separated, image):
        left,right=separated
        height=image.shape[0]
        lanes=[]

        # 오른쪽
        if right:
            x,y=[],[]
            for line in right:
                x1,y1,x2,y2=line[0]
                x.extend([x1,x2])
                y.extend([y1,y2])
            poly=np.polyfit(y,x,1)
            f=np.poly1d(poly)
            y1_frame,y2_frame=height,int(height*0.55)
            lanes.append(((int(f(y1_frame)),y1_frame),(int(f(y2_frame)),y2_frame)))
            self.right_m=poly[0]
            self.right_b=(poly[1],0)
        else:
            lanes.append(self.prev_lanes[0])  # 이전 프레임 사용

        # 왼쪽
        if left:
            x,y=[],[]
            for line in left:
                x1,y1,x2,y2=line[0]
                x.extend([x1,x2])
                y.extend([y1,y2])
            poly=np.polyfit(y,x,1)
            f=np.poly1d(poly)
            y1_frame,y2_frame=height,int(height*0.55)
            lanes.append(((int(f(y1_frame)),y1_frame),(int(f(y2_frame)),y2_frame)))
            self.left_m=poly[0]
            self.left_b=(poly[1],0)
        else:
            lanes.append(self.prev_lanes[1])

        self.prev_lanes=lanes  # 이전 프레임 저장
        return lanes

    # 방향 예측
    def predictDir(self):
        return "Straight"

    # 강화된 lane change 감지
    def detect_lane_change(self, lanes, frame):
        height,width=frame.shape[:2]
        car_center_x=width//2
        right_lane,left_lane=lanes

        if right_lane and left_lane:
            lane_center=(right_lane[0][0]+left_lane[0][0])//2
        elif right_lane:
            lane_center=right_lane[0][0]-50
        elif left_lane:
            lane_center=left_lane[0][0]+50
        else:
            self.change_counter=0
            return "Lane Not Detected"

        if car_center_x<lane_center-self.margin:
            current_state="left"
        elif car_center_x>lane_center+self.margin:
            current_state="right"
        else:
            current_state="center"

        if current_state!=self.prev_lane_state:
            self.change_counter+=1
            if self.change_counter>=self.threshold_frames:
                self.prev_lane_state=current_state
                self.change_counter=0
                return "Lane Change Detected!"
        else:
            self.change_counter=0
        return "Normal Driving"

    # 차선 영역과 결과 시각화
    def draw_result(self, frame, lanes, direction, status):
        overlay=frame.copy()
        output=frame.copy()
        right_lane,left_lane=lanes

        # 다각형 좌표 구성
        poly_points=[]
        if left_lane and right_lane:
            poly_points=[left_lane[0],left_lane[1],right_lane[1],right_lane[0]]
            cv2.fillConvexPoly(overlay,np.array(poly_points,np.int32),(0,200,0))
            cv2.addWeighted(overlay,0.3,output,0.7,0,output)

        # 오른쪽 분홍, 왼쪽 노랑
        if right_lane:
            cv2.line(output,right_lane[0],right_lane[1],(255,0,255),5)
        if left_lane:
            cv2.line(output,left_lane[0],left_lane[1],(0,255,255),5)

        # 텍스트
        cv2.putText(output,f"Dir: {direction}",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        cv2.putText(output,f"Status: {status}",(50,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        return output
        
    def detect_lanes(self, frame):
        """
        frame -> 좌/우 차선 x좌표 반환
        없으면 None 반환
        """
        filtered = self.filter_colors(frame)
        edges = cv2.Canny(filtered, 100, 200)
        roi = self.limit_region(edges)
        lines = self.houghLines(roi)
        if lines is None or len(lines) == 0:
            return None

        separated = self.separateLine(roi, lines)
        lanes = self.regression(separated, frame)
        if lanes is None or len(lanes) < 4:
            return None

        # lanes[2]=왼쪽 아래, lanes[0]=오른쪽 아래
        left_x = lanes[2][0] if lanes[2] is not None else 0
        right_x = lanes[0][0] if lanes[0] is not None else frame.shape[1]

        return (left_x, right_x)
