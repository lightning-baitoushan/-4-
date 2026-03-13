#include <ESP8266WiFi.h>
#include <Servo.h>

// ===================== WiFi配置 =====================
const char* ssid     = "你的手机热点名称";
const char* password = "你的手机热点密码";

// ===================== TCP服务器配置 =====================
WiFiServer server(8080);
WiFiClient client;

// ===================== 舵机配置 =====================
// 舵机引脚定义（NodeMCU: D1..D4）
static const uint8_t SERVO_PINS[4] = { D1, D2, D3, D4 }; // 底座, 前臂, 后臂, 钳子

// 舵机角度范围
static const int SERVO_MIN[4] = { 0, 45, 30, 40 };
static const int SERVO_MAX[4] = { 180, 135, 150, 120 };

static const char* SERVO_NAME[4] = { "BASE", "FOREARM", "BACKARM", "CLAMP" };

// Servo库脉宽范围（与原代码 500~2500us 保持一致）
static const int SERVO_PULSE_MIN_US = 500;
static const int SERVO_PULSE_MAX_US = 2500;

// 非阻塞平滑移动参数（关键：loop里按周期推进，不在网络处理里delay）
static const uint16_t SERVO_UPDATE_INTERVAL_MS = 20;  // 典型舵机刷新周期 20ms
static const float    SERVO_STEP_DEG = 2.0f;          // 每次刷新最多走的角度（2°@20ms≈100°/s）

Servo servos[4];

// 当前角度 / 目标角度（用float避免累计误差；写入舵机时取整）
float currentAngles[4] = { 90, 90, 90, 90 };
float targetAngles[4]  = { 90, 90, 90, 90 };
int   lastWritten[4]   = { -1, -1, -1, -1 };

unsigned long lastServoUpdateMs = 0;

// ===================== 行缓冲（非阻塞收包） =====================
static char lineBuf[64];
static uint8_t linePos = 0;

// ===================== WiFi重连（非阻塞） =====================
static unsigned long lastWiFiAttemptMs = 0;
static const unsigned long WIFI_RETRY_INTERVAL_MS = 5000;

static void connectToWiFiBlocking(unsigned long timeoutMs) {
  Serial.print("连接到WiFi: ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.begin(ssid, password);

  const unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < timeoutMs) {
    delay(250);
    Serial.print(".");
    yield();
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi连接成功");
    Serial.print("IP地址: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi连接超时（稍后在loop里自动重试）");
  }
}

static void ensureWiFiNonBlocking() {
  if (WiFi.status() == WL_CONNECTED) return;

  unsigned long now = millis();
  if (now - lastWiFiAttemptMs < WIFI_RETRY_INTERVAL_MS) return;

  lastWiFiAttemptMs = now;
  Serial.println("WiFi断开，尝试重连...");
  WiFi.disconnect();
  WiFi.begin(ssid, password);
}

// 解析命令 "a1,a2,a3,a4"
static bool parseAngles(const char* line, int outAngles[4]) {
  // 复制到本地缓冲，避免strtok修改原数据
  char buf[64];
  strncpy(buf, line, sizeof(buf));
  buf[sizeof(buf) - 1] = '\0';

  char* token = strtok(buf, ",");
  int idx = 0;

  while (token != nullptr && idx < 4) {
    outAngles[idx++] = atoi(token);
    token = strtok(nullptr, ",");
  }

  return (idx == 4);
}

static void applyTargets(const int inAngles[4]) {
  for (int i = 0; i < 4; i++) {
    int v = inAngles[i];
    v = constrain(v, SERVO_MIN[i], SERVO_MAX[i]);
    targetAngles[i] = (float)v;
  }

  Serial.print("目标角度: ");
  for (int i = 0; i < 4; i++) {
    Serial.print((int)targetAngles[i]);
    if (i < 3) Serial.print(",");
  }
  Serial.println();
}

// 非阻塞平滑更新（每20ms推进一次，不阻塞网络读包）
static void updateServosNonBlocking() {
  unsigned long now = millis();
  if (now - lastServoUpdateMs < SERVO_UPDATE_INTERVAL_MS) return;
  lastServoUpdateMs = now;

  for (int i = 0; i < 4; i++) {
    float cur = currentAngles[i];
    float tgt = targetAngles[i];
    float delta = tgt - cur;

    if (fabs(delta) < 0.001f) continue;

    if (delta > SERVO_STEP_DEG) {
      cur += SERVO_STEP_DEG;
    } else if (delta < -SERVO_STEP_DEG) {
      cur -= SERVO_STEP_DEG;
    } else {
      cur = tgt;
    }

    // 夹紧到合法范围
    if (cur < SERVO_MIN[i]) cur = (float)SERVO_MIN[i];
    if (cur > SERVO_MAX[i]) cur = (float)SERVO_MAX[i];

    currentAngles[i] = cur;

    int writeAngle = (int)lround(cur);
    if (writeAngle != lastWritten[i]) {
      servos[i].write(writeAngle);
      lastWritten[i] = writeAngle;
    }
  }
}

static void acceptClientIfNeeded() {
  if (client && client.connected()) return;

  if (client) client.stop();

  WiFiClient newClient = server.available();
  if (!newClient) return;

  client = newClient;
  client.setNoDelay(true);
  linePos = 0;
  Serial.println("新客户端连接");
}

static void readClientNonBlocking() {
  if (!client || !client.connected()) return;

  while (client.available()) {
    char c = (char)client.read();
    if (c == '\r') continue;

    if (c == '\n') {
      lineBuf[linePos] = '\0';

      if (linePos > 0) {
        int angles[4];
        if (parseAngles(lineBuf, angles)) {
          applyTargets(angles);
          client.println("OK");
        } else {
          Serial.print("命令格式错误: ");
          Serial.println(lineBuf);
          client.println("ERR");
        }
      }

      linePos = 0;
      continue;
    }

    if (linePos < sizeof(lineBuf) - 1) {
      lineBuf[linePos++] = c;
    } else {
      // 溢出保护：丢弃这一行
      linePos = 0;
    }
  }
}

void setup() {
  Serial.begin(9600);
  delay(200);

  // 初始化舵机
  for (int i = 0; i < 4; i++) {
    servos[i].attach(SERVO_PINS[i], SERVO_PULSE_MIN_US, SERVO_PULSE_MAX_US);
    currentAngles[i] = targetAngles[i] = 90.0f;
    servos[i].write(90);
    lastWritten[i] = 90;
  }

  // 连接WiFi（setup里可以阻塞一次）
  connectToWiFiBlocking(15000);

  // 启动TCP服务器
  server.begin();
  Serial.println("TCP服务器已启动");
  Serial.print("服务器地址: ");
  Serial.println(WiFi.localIP());
  Serial.println("端口: 8080");
}

void loop() {
  ensureWiFiNonBlocking();

  acceptClientIfNeeded();
  readClientNonBlocking();

  updateServosNonBlocking();

  // 让出CPU给WiFi协议栈（很重要）
  yield();
}