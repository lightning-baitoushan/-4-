 #include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>

// ==================== 配置 ====================
// WiFi配置 - 手机热点
const char* ssid = "REDMI K80";        // 修改为你的热点名称
const char* password = "mdwjgkuurmyenjf";  // 修改为你的热点密码

// TCP服务器端口
const int tcpPort = 8888;

// 引脚定义
#define LED_PIN 2          // 板载LED
#define RESET_PIN 0         // 复位按钮

// ==================== 全局变量 ====================
WiFiServer server(tcpPort);
WiFiClient client;
ESP8266WebServer webServer(80);

// 连接状态
bool clientConnected = false;
unsigned long lastHeartbeat = 0;
const unsigned long heartbeatInterval = 3000;  // 3秒心跳

// 舵机状态
struct ServoStatus {
  int currentAngle;
  int targetAngle;
};

ServoStatus baseServo = {90, 90};
ServoStatus clawServo = {45, 45};
ServoStatus forearmServo = {60, 60};
ServoStatus reararmServo = {60, 60};

// 命令队列
#define QUEUE_SIZE 30
String cmdQueue[QUEUE_SIZE];
int queueHead = 0;
int queueTail = 0;

// 统计
unsigned long cmdCount = 0;
unsigned long lastStatusTime = 0;

// ==================== 函数声明 ====================
void connectWiFi();
void handleClient();
void processCommand(String cmd);
void sendToArduino(String cmd);
void readFromArduino();
void sendStatus();
void blinkLED(int times, int delayMs);
void resetServos();
void setupWebServer();

// ==================== 初始化 ====================
void setup() {
  // 初始化串口
  Serial.begin(74880);  // 调试串口
  Serial.setTimeout(50);
  
  Serial1.begin(115200);  // 与Arduino通信的串口
  Serial1.setTimeout(50);
  
  // 初始化引脚
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);  // 关闭LED
  
  pinMode(RESET_PIN, INPUT_PULLUP);
  
  // 打印启动信息
  Serial.println("\n\n=================================");
  Serial.println("ESP8266 Robot Controller v3.0");
  Serial.println("=================================");
  
  // 连接WiFi
  connectWiFi();
  
  // 启动TCP服务器
  server.begin();
  Serial.printf("TCP服务器启动，端口: %d\n", tcpPort);
  
  // 设置Web服务器
  setupWebServer();
  
  // 启动心跳
  lastHeartbeat = millis();
  
  // 启动提示
  blinkLED(3, 200);
  digitalWrite(LED_PIN, LOW);  // 常亮表示就绪
  
  Serial.println("系统初始化完成");
  Serial.printf("IP地址: %s\n", WiFi.localIP().toString().c_str());
  Serial.println("等待Python客户端连接...");
}

// ==================== WiFi连接 ====================
void connectWiFi() {
  Serial.printf("连接WiFi: %s\n", ssid);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  }
  
  Serial.println();
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✅ WiFi连接成功");
    Serial.printf("IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("❌ WiFi连接失败，启动AP模式");
    WiFi.softAP("Robot_AP", "12345678");
    Serial.printf("AP IP: %s\n", WiFi.softAPIP().toString().c_str());
  }
}

// ==================== Web服务器 ====================
void setupWebServer() {
  webServer.on("/", []() {
    String html = "<!DOCTYPE html><html>";
    html += "<head><meta charset='UTF-8'><meta http-equiv='refresh' content='2'>";
    html += "<style>body{font-family:Arial;margin:20px;background:#f0f0f0;}";
    html += ".container{max-width:600px;margin:auto;background:white;padding:20px;border-radius:10px;}";
    html += ".ok{color:green;}.error{color:red;}</style></head><body>";
    html += "<div class='container'>";
    html += "<h2>🤖 机器人控制器状态</h2>";
    
    // WiFi状态 - 修复字符串拼接
    html += "<p>WiFi: <span class='";
    if (WiFi.status() == WL_CONNECTED) {
      html += "ok'>已连接";
    } else {
      html += "error'>未连接";
    }
    html += "</span></p>";
    
    html += "<p>IP地址: " + WiFi.localIP().toString() + "</p>";
    
    // 客户端状态 - 修复字符串拼接
    html += "<p>Python客户端: <span class='";
    if (clientConnected) {
      html += "ok'>已连接";
    } else {
      html += "error'>未连接";
    }
    html += "</span></p>";
    
    // 舵机状态
    html += "<h3>舵机角度</h3>";
    html += "<p>底座: " + String(baseServo.currentAngle) + "° / " + String(baseServo.targetAngle) + "°</p>";
    html += "<p>钳子: " + String(clawServo.currentAngle) + "° / " + String(clawServo.targetAngle) + "°</p>";
    html += "<p>小臂: " + String(forearmServo.currentAngle) + "° / " + String(forearmServo.targetAngle) + "°</p>";
    html += "<p>大臂: " + String(reararmServo.currentAngle) + "° / " + String(reararmServo.targetAngle) + "°</p>";
    
    // 统计
    html += "<p>命令计数: " + String(cmdCount) + "</p>";
    html += "<p>运行时间: " + String(millis() / 1000) + "秒</p>";
    
    html += "</div></body></html>";
    webServer.send(200, "text/html", html);
  });
  
  webServer.on("/reset", []() {
    resetServos();
    webServer.send(200, "text/plain", "OK");
  });
  
  webServer.begin();
  Serial.println("Web服务器已启动");
}

// ==================== 重置舵机 ====================
void resetServos() {
  Serial.println("重置所有舵机");
  
  baseServo.targetAngle = 90;
  clawServo.targetAngle = 45;
  forearmServo.targetAngle = 60;
  reararmServo.targetAngle = 60;
  
  // 发送重置命令到Arduino
  Serial1.println("RESET:0");
  
  blinkLED(5, 100);
}

// ==================== 主循环 ====================
void loop() {
  // 处理TCP客户端
  handleClient();
  
  // 处理Web请求
  webServer.handleClient();
  
  // 读取Arduino数据
  readFromArduino();
  
  // 处理命令队列
  if (queueHead != queueTail) {
    String cmd = cmdQueue[queueTail];
    queueTail = (queueTail + 1) % QUEUE_SIZE;
    sendToArduino(cmd);
  }
  
  // 心跳
  if (millis() - lastHeartbeat > heartbeatInterval) {
    lastHeartbeat = millis();
    
    // 闪烁LED
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    
    // 发送心跳到客户端
    if (clientConnected && client.connected()) {
      client.println("HEARTBEAT");
    }
    
    // 请求Arduino状态
    Serial1.println("STATUS?");
    
    // 定期发送状态到客户端
    if (clientConnected && client.connected()) {
      sendStatus();
    }
  }
  
  // 检查复位按钮
  if (digitalRead(RESET_PIN) == LOW) {
    delay(50);
    if (digitalRead(RESET_PIN) == LOW) {
      resetServos();
      while (digitalRead(RESET_PIN) == LOW) {
        delay(10);
      }
    }
  }
  
  delay(10);
}

// ==================== 处理TCP客户端 ====================
void handleClient() {
  // 检查新连接
  if (!clientConnected) {
    client = server.available();
    if (client) {
      if (client.connected()) {
        clientConnected = true;
        Serial.println("✅ Python客户端已连接");
        client.println("ROBOT_CONTROLLER_READY");
        blinkLED(2, 100);
      }
    }
  }
  
  // 处理已连接的客户端
  if (clientConnected && client.connected()) {
    if (client.available()) {
      String data = client.readStringUntil('\n');
      data.trim();
      
      if (data.length() > 0) {
        Serial.printf("📥 从Python: %s\n", data.c_str());
        
        // 处理命令
        processCommand(data);
        
        // 发送确认
        client.println("ACK:" + data);
      }
    }
  } else {
    if (clientConnected) {
      clientConnected = false;
      Serial.println("❌ Python客户端断开");
    }
  }
}

// ==================== 处理命令 ====================
void processCommand(String cmd) {
  cmd.toUpperCase();
  
  if (cmd == "RESET:0") {
    resetServos();
    return;
  }
  
  // 解析命令格式: SERVO:DIRECTION:ANGLE
  int firstColon = cmd.indexOf(':');
  int secondColon = cmd.indexOf(':', firstColon + 1);
  
  if (firstColon == -1 || secondColon == -1) {
    Serial.println("命令格式错误");
    return;
  }
  
  String servo = cmd.substring(0, firstColon);
  String direction = cmd.substring(firstColon + 1, secondColon);
  int angle = cmd.substring(secondColon + 1).toInt();
  
  if (angle <= 0) angle = 1;
  if (angle > 180) angle = 180;
  
  // 添加到队列
  int nextHead = (queueHead + 1) % QUEUE_SIZE;
  if (nextHead != queueTail) {
    cmdQueue[queueHead] = servo + ":" + direction + ":" + String(angle);
    queueHead = nextHead;
    cmdCount++;
    Serial.printf("命令加入队列 [%d]: %s\n", cmdCount, cmdQueue[(queueHead - 1 + QUEUE_SIZE) % QUEUE_SIZE].c_str());
  } else {
    Serial.println("⚠️ 命令队列已满");
  }
}

// ==================== 发送到Arduino ====================
void sendToArduino(String cmd) {
  Serial.printf("📤 到Arduino: %s\n", cmd.c_str());
  Serial1.println(cmd);
  
  // 更新目标角度
  int firstColon = cmd.indexOf(':');
  int secondColon = cmd.indexOf(':', firstColon + 1);
  
  String servo = cmd.substring(0, firstColon);
  String direction = cmd.substring(firstColon + 1, secondColon);
  int angle = cmd.substring(secondColon + 1).toInt();
  
  ServoStatus* target = nullptr;
  if (servo == "BASE") target = &baseServo;
  else if (servo == "CLAW") target = &clawServo;
  else if (servo == "FOREARM") target = &forearmServo;
  else if (servo == "REARARM") target = &reararmServo;
  
  if (target) {
    // 直接使用当前角度作为目标角度，因为Arduino会处理相对运动
    // 这样可以避免角度值被调整两次的问题
    if (direction == "CW") {
      target->targetAngle = constrain(target->currentAngle + angle, 0, 180);
    } else {
      target->targetAngle = constrain(target->currentAngle - angle, 0, 180);
    }
    
    // 限制范围
    if (servo == "BASE") target->targetAngle = constrain(target->targetAngle, 0, 180);
    else if (servo == "CLAW") target->targetAngle = constrain(target->targetAngle, 0, 90);
    else if (servo == "FOREARM") target->targetAngle = constrain(target->targetAngle, 0, 135);
    else if (servo == "REARARM") target->targetAngle = constrain(target->targetAngle, 0, 120);
  }
}

// ==================== 读取Arduino数据 ====================
void readFromArduino() {
  while (Serial1.available()) {
    String data = Serial1.readStringUntil('\n');
    data.trim();
    
    if (data.length() > 0) {
      Serial.printf("📥 从Arduino: %s\n", data.c_str());
      
      // 解析状态
      if (data.startsWith("STATUS:")) {
        // STATUS:BASE:90:CLAW:45:FOREARM:60:REARARM:60
        data = data.substring(7);
        
        int p1 = data.indexOf(':');
        int p2 = data.indexOf(':', p1 + 1);
        int p3 = data.indexOf(':', p2 + 1);
        int p4 = data.indexOf(':', p3 + 1);
        int p5 = data.indexOf(':', p4 + 1);
        int p6 = data.indexOf(':', p5 + 1);
        int p7 = data.indexOf(':', p6 + 1);
        
        if (p1 > 0 && p2 > 0 && p3 > 0 && p4 > 0 && p5 > 0 && p6 > 0 && p7 > 0) {
          baseServo.currentAngle = data.substring(p1 + 1, p2).toInt();
          clawServo.currentAngle = data.substring(p3 + 1, p4).toInt();
          forearmServo.currentAngle = data.substring(p5 + 1, p6).toInt();
          reararmServo.currentAngle = data.substring(p7 + 1).toInt();
        }
      }
    }
  }
}

// ==================== 发送状态到客户端 ====================
void sendStatus() {
  if (clientConnected && client.connected()) {
    String status = "STATUS:";
    status += "BASE:" + String(baseServo.currentAngle) + ":";
    status += "CLAW:" + String(clawServo.currentAngle) + ":";
    status += "FOREARM:" + String(forearmServo.currentAngle) + ":";
    status += "REARARM:" + String(reararmServo.currentAngle);
    client.println(status);
  }
}

// ==================== LED闪烁 ====================
void blinkLED(int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
  }
}