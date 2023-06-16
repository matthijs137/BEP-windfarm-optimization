#include <AccelStepper.h>

//Sensor pins
const int sensor1_pin = A0;
const int sensor2_pin = A1;
const int sensor3_pin = A2;

//connect the pins
const int stepPin1 = 3;
const int dirPin1 = 2;
const int stepPin2 = 5;
const int dirPin2 = 4;
const int stepPin3 = 7;
const int dirPin3 = 6;

const int StepsPerRev = 400;

//Make Accelstepper objects
AccelStepper stepper1(AccelStepper::DRIVER, stepPin1, dirPin1);
AccelStepper stepper2(AccelStepper::DRIVER, stepPin2, dirPin2);
AccelStepper stepper3(AccelStepper::DRIVER, stepPin3, dirPin3);

// Define the variables for the amplitudes, phase and frequencys
float A_one = 0;
float A_two = 0;
float P = 0;
float f_one = 0;
float f_two = 0;

float sensor1value = 0;
float sensor2value = 0;
float sensor3value = 0;

// Define the variable for controlling the test
bool test_running = false;

//Define a personal time variable
unsigned long myMillis = 0;

void setup()
{
  // Initialize the serial communication
  Serial.begin(115200);

  // Stepper settings
  stepper1.setMaxSpeed(8000);
  stepper1.setAcceleration(8000);
  stepper1.setCurrentPosition(0);

  stepper2.setMaxSpeed(8000);
  stepper2.setAcceleration(8000);
  stepper2.setCurrentPosition(0);

  stepper3.setMaxSpeed(8000);
  stepper3.setAcceleration(8000);
  stepper3.setCurrentPosition(0);

  pinMode(sensor1_pin, INPUT);
  pinMode(sensor2_pin, INPUT);
  pinMode(sensor3_pin, INPUT);
}

void loop()
{
  // Check if there is any incoming serial data
  if (Serial.available() > 0)
  {
    // Read the incoming command
    String command = Serial.readStringUntil('\n');


    if (command == "stop")
    {
      // Set the test_running flag to false and reset the stepper angles
      test_running = false;
      
      stepper1.moveTo(0);
      stepper2.moveTo(0);
      stepper3.moveTo(0);
      stepper1.runToPosition();
      stepper2.runToPosition();
      stepper3.runToPosition();
    }
    else if ((command.startsWith("start:") && (test_running == false)))
    {
      // Parse the values from the command
      // Start by removing the "start:" prefix
      command = command.substring(6);
      
      int index_one = command.indexOf(",");
      A_one = command.substring(0, index_one).toFloat();
      String rest_one = command.substring(index_one + 1);

      int index_two = rest_one.indexOf(";");
      A_two = rest_one.substring(0, index_two).toFloat();
      String rest_two = rest_one.substring(index_two + 1);

      int index_three = rest_two.indexOf("-");
      P = rest_two.substring(0, index_three).toFloat();
      String rest_three = rest_two.substring(index_three + 1);

      int index_four = rest_three.indexOf("+");
      f_one = rest_three.substring(0, index_four).toFloat();
      f_two = rest_three.substring(index_four + 1).toFloat();
      
      test_running = true;
      myMillis = millis();
    }
  }

  // Check if the test is running
  if (test_running)
  {
   // Calculate the current angles based on the amplitudes, phase and frequencys
   float angle_one = 0.0 +  (A_one * sin(2 * PI * f_one * (millis() - myMillis) / 1000.0));
   float angle_two = 0.0 + (A_two * sin(2 * PI * f_two * (millis() - myMillis - ((P/360)*(1/f_two)*1000)) / 1000.0));
   float angle_three = 0.0;

   //Read sensors
   sensor1value = analogRead(sensor1_pin);
   sensor2value = analogRead(sensor2_pin);
   sensor3value = analogRead(sensor3_pin);
   
   //Send data back to Python
   String p1 = ";";
   Serial.println((millis() - myMillis) + p1 + angle_one + p1 + angle_two + p1  + sensor1value + p1 + sensor2value + p1 + sensor3value);

   // Set the steppers angles to the current angle
   int steps_one = (angle_one/360.0) * StepsPerRev;
   int steps_two = (angle_two/360.0) * StepsPerRev;
   int steps_three = (angle_three/360.0)*StepsPerRev;
   
   stepper1.moveTo(steps_one);
   stepper2.moveTo(steps_two);
   stepper3.moveTo(steps_three);
   stepper1.run();
   stepper2.run();
   stepper3.run();

  }
}
