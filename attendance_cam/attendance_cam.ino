#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>

// ---------- WIFI CREDENTIALS ----------
const char* ssid = "iPhone";
const char* password = "dinesh@123";

// ---------- PIN DEFINITIONS ----------
#define SUBJECT1_PIN 26 
#define SUBJECT2_PIN 27
#define START_PIN 32
#define STOP_PIN 33 
#define SDA_PIN 25
#define SCL_PIN 21 

// ---------- LCD ----------
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ---------- WEB SERVER ----------
WebServer server(80);

// ---------- BUTTON EVENT FLAGS ----------
bool subject1Event = false;
bool subject2Event = false;
bool startEvent = false;
bool stopEvent = false;

// ---------- TRACK LAST BUTTON STATES ----------
int lastSubject1State = HIGH;
int lastSubject2State = HIGH;
int lastStartState = HIGH;
int lastStopState = HIGH;

// ---------- TRACK CURRENT SUBJECT ----------
int currentSubject = 0; // 0=none, 1=subject1, 2=subject2

// ---------- HANDLE /buttoninfo ----------
void handleButtonInfo() {
  DynamicJsonDocument doc(256);
  
  // Send current event states
  doc["subject1"] = subject1Event ? 1 : 0;
  doc["subject2"] = subject2Event ? 1 : 0;
  doc["start"] = startEvent ? 1 : 0;
  doc["stop"] = stopEvent ? 1 : 0;

  String jsonResponse;
  serializeJson(doc, jsonResponse);
  
  Serial.println("Button info requested:");
  Serial.println(jsonResponse);
  
  server.send(200, "application/json", jsonResponse);

  // Reset events AFTER Python reads them
  subject1Event = false;
  subject2Event = false;
  startEvent = false;
  stopEvent = false;
}

// ---------- HANDLE /displayinfo ----------
void handleDisplayInfo() {
  if (server.hasArg("plain") == false) {
    Serial.println("Error: No data received");
    server.send(400, "text/plain", "Bad Request: No data received");
    return;
  }

  String message = server.arg("plain");
  Serial.println("Received display request:");
  Serial.println(message);
  
  DynamicJsonDocument doc(256);
  DeserializationError error = deserializeJson(doc, message);

  if (error) {
    Serial.println("Error: Invalid JSON");
    server.send(400, "text/plain", "Bad Request: Invalid JSON");
    return;
  }

  if (doc.containsKey("message")) {
    String displayMessage = doc["message"].as<String>();
    Serial.print("Displaying on LCD: ");
    Serial.println(displayMessage);
    
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print(displayMessage.substring(0, 16));
    lcd.setCursor(0, 1);
    lcd.print(displayMessage.substring(16, 32));
    
    server.send(200, "text/plain", "Message displayed on LCD");
  } else {
    Serial.println("Error: 'message' key missing");
    server.send(400, "text/plain", "Bad Request: 'message' key missing");
  }
}

// ---------- CHECK BUTTON PRESSES ----------
void checkButtons() {
  // Read current button states
  int subject1Reading = digitalRead(SUBJECT1_PIN);
  int subject2Reading = digitalRead(SUBJECT2_PIN);
  int startReading = digitalRead(START_PIN);
  int stopReading = digitalRead(STOP_PIN);

  // ✅ FIX: Detect button press (transition from HIGH to LOW)
  // Subject 1 button
  if (subject1Reading == LOW && lastSubject1State == HIGH) {
    delay(50); // Simple debounce
    subject1Reading = digitalRead(SUBJECT1_PIN); // Re-read after debounce
    if (subject1Reading == LOW) { // Confirm still pressed
      subject1Event = true;
      currentSubject = 1;
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Subject1 pressed");
      lcd.setCursor(0, 1);
      lcd.print("Press Start");
      Serial.println("✅ Subject 1 pressed - Event flag set");
    }
  }
  lastSubject1State = subject1Reading;

  // Subject 2 button
  if (subject2Reading == LOW && lastSubject2State == HIGH) {
    delay(50);
    subject2Reading = digitalRead(SUBJECT2_PIN);
    if (subject2Reading == LOW) {
      subject2Event = true;
      currentSubject = 2;
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Subject2 pressed");
      lcd.setCursor(0, 1);
      lcd.print("Press Start");
      Serial.println("✅ Subject 2 pressed - Event flag set");
    }
  }
  lastSubject2State = subject2Reading;

  // Start button
  if (startReading == LOW && lastStartState == HIGH) {
    delay(50);
    startReading = digitalRead(START_PIN);
    if (startReading == LOW) {
      if (currentSubject > 0) {
        startEvent = true;
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Attendance");
        lcd.setCursor(0, 1);
        lcd.print("Started");
        Serial.println("✅ Start button pressed - Event flag set");
      } else {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Select Subject");
        lcd.setCursor(0, 1);
        lcd.print("First!");
        Serial.println("⚠️ Start pressed without subject selection");
        delay(1500);
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("Press Any");
        lcd.setCursor(0, 1);
        lcd.print("Subject Button");
      }
    }
  }
  lastStartState = startReading;

  // Stop button
  if (stopReading == LOW && lastStopState == HIGH) {
    delay(50);
    stopReading = digitalRead(STOP_PIN);
    if (stopReading == LOW) {
      stopEvent = true;
      currentSubject = 0;
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Attendance");
      lcd.setCursor(0, 1);
      lcd.print("Stopped");
      Serial.println("✅ Stop button pressed - Event flag set");
      delay(2000);
      lcd.clear();
      lcd.setCursor(0, 0);
      lcd.print("Press Any");
      lcd.setCursor(0, 1);
      lcd.print("Subject Button");
    }
  }
  lastStopState = stopReading;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n=== ESP32 Attendance System ===");

  pinMode(SUBJECT1_PIN, INPUT_PULLUP);
  pinMode(SUBJECT2_PIN, INPUT_PULLUP);
  pinMode(START_PIN, INPUT_PULLUP);
  pinMode(STOP_PIN, INPUT_PULLUP);

  Wire.begin(SDA_PIN, SCL_PIN);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("Attendance Sys");
  lcd.setCursor(0, 1);
  lcd.print("Starting...");
  Serial.println("LCD initialized");
  delay(2000);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Press Any");
  lcd.setCursor(0, 1);
  lcd.print("Subject Button");

  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi Connected!");
    Serial.print("ESP32 IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.println("Web server started on port 80");
    Serial.println("Endpoints:");
    Serial.println("  GET  /buttoninfo");
    Serial.println("  POST /displayinfo");
  } else {
    Serial.println("\n❌ WiFi connection failed!");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("WiFi Failed!");
  }

  server.on("/buttoninfo", handleButtonInfo);
  server.on("/displayinfo", HTTP_POST, handleDisplayInfo);
  server.begin();
  
  Serial.println("Ready! Press buttons to test...");
}

void loop() {
  server.handleClient();
  checkButtons();
}
