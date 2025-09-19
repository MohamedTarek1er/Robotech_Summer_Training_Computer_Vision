#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

String server = "broker.hivemq.com";
String ssid = "Modo";
String pass = "Doha_bio20";

WiFiClient wifiClient;
PubSubClient client(wifiClient);
String clientID = "HandGestureESP";
String topic = "meow/psps";

void setup() 
{
  Serial.begin(115200);
  Serial.println("Hello, ESP8266!");

  Wire.begin(D2, D1); // SDA=D2, SCL=D1
  lcd.init();
  lcd.backlight();
  
  lcd.setCursor(0, 0);
  Serial.println("LCD ON!");
  lcd.clear();
  lcd.print("LCD ON");
  Serial.println("Printed on LCD!");
  delay(5000);
  lcd.clear();

  wifiSetup();
  client.setServer(server.c_str(), 1883);
  client.setCallback(callback);
}

void loop() 
{
  if (!client.connected())
    reconnect();

  client.loop();
}

void wifiSetup()
{
  Serial.println("Attempting to connect to WiFi!");
  WiFi.begin(ssid.c_str(), pass.c_str());
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(100);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi!");
}

void callback(const char* topic, byte* payload, int size)
{
  String message = "";
  for (int i = 0; i < size; i++)
    message += (char)payload[i];

  lcd.clear();
  lcd.print(message);
}

void reconnect()
{
  Serial.println("Attempting to connect to server!");
  while (!client.connected())
  {
    if (client.connect(clientID.c_str()))
    {
      client.subscribe(topic.c_str());
      break;
    }
  }
  Serial.println("Server Connection successful!\nSubscribed!");
}
