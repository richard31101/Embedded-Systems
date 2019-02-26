#include <Wire.h>
#include "MAX30105.h"
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include "SparkFunLIS3DH.h"
#include "SPI.h"

LIS3DH myIMU; //Default constructor is I2C, addr 0x19.
MAX30105 particleSensor;
String activity[6] = {"Sleeping","Sitting","Standing","Walking","Jogging","Running"};
int powerLevel = 0x1F;   //6.4mA
int count = 0;
int accI=0; // int i=0~5, meaning each activity sleeping, sitting, standing, walking, jogging, running
String data = "";
unsigned long startTime;
// WiFi settings
const char *ssid = "UCInet Mobile Access";

void connectWifi()
{
    WiFi.begin(ssid);//, wifipasswd);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    // Print the IP address
    Serial.println(WiFi.localIP());
}

void postServer(String postMesg, int activity)
{
    String data;
    data = "data={\"activity\": "+String(activity)+", \"data\": ["+postMesg+"]}";

    if (WiFi.status() != WL_CONNECTED){
        Serial.println("WIFI Error");
        return;
    }
        HTTPClient http;
        //Serial.print("[HTTP] begin...\n");
        http.begin("http://13.58.170.45/cs244hw8.php/"); //HTTP
        
        digitalWrite(LED_BUILTIN, LOW);
        //Serial.print("[HTTP] POST...\n");

        http.addHeader("Content-Type", "application/x-www-form-urlencoded");
        //Serial.println(data);        
        int httpCode = http.POST(data); 

        if(httpCode > 0) {
            //Serial.printf("[HTTP] POST... code: %d\n", httpCode);
            if(httpCode == HTTP_CODE_OK) {
                String payload = http.getString();
                //Serial.println(payload);
            }
        } else {
            Serial.printf("[HTTP] POST... failed, error: %s\n", http.errorToString(httpCode).c_str());
        }
        http.end();
        digitalWrite(LED_BUILTIN, HIGH);
}


void startPPGSensor(int powerLevel)
{
    // Initialize sensor
    if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) //Use default I2C port, 400kHz speed
    {
      Serial.println("MAX30105 was not found. Please check wiring/power. ");
      while (1);
    }
    byte sampleAverage = 1; // No averaging of samples
    byte ledMode = 2; //Red and IR
    int sampleRate = 50; // sample frequency = 50 Hz
    int pulseWidth = 100;
    int adcRange = 16384; // default (62.5pA per LSB)
    particleSensor.setup(powerLevel, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
}


void setup()
{
    Serial.begin(115200);
    //Serial.println("Initializing...");
    connectWifi();
    startPPGSensor(powerLevel); 
    myIMU.begin(); // start to capture data from Accelerometer sensor
}

void loop()
{
    //Serial.print("\n*********");
    //Serial.print(activity[accI]);
    //Serial.print("*********\n");
    //startTime = millis(); 

    data += "[";
    data += particleSensor.getIR(); // post the updated IR reading
    data += ",";
    data += particleSensor.getRed(); // post the updated Red reading
    data += ",";
    data += myIMU.readRawAccelX(); // post the updated accelerometer axis X reading
    data += ",";
    data += myIMU.readRawAccelY(); // post the updated accelerometer axis Y reading
    data += ",";
    data += myIMU.readRawAccelZ(); // post the updated accelerometer axis Z reading
    data += "]";
    /*
    Serial.print("\nIR[");
    Serial.print(particleSensor.getIR()); // show the updated IR reading in the terminal 
    Serial.print("] RED[");
    Serial.print(particleSensor.getRed()); // show the updated RED reading in the terminal
    Serial.print("]\n");  
    Serial.print("\nAccelerometer:\n");
    Serial.print(" X = ");
    Serial.println(myIMU.readFloatAccelX(), 4);
    Serial.print(" Y = ");
    Serial.println(myIMU.readFloatAccelY(), 4);
    Serial.print(" Z = ");
    Serial.println(myIMU.readFloatAccelZ(), 4);
    Serial.println();
    */
    count++;
    if (count == 80){      //post to server after every 50 [IR, RED, AccX, AccY, AccZ] data tuples captured
        postServer(data,accI);    //accI is for activity index, for seperating different cvs
        count = 0;
        data = "";
    } else {
        data += ",";
    }     
}