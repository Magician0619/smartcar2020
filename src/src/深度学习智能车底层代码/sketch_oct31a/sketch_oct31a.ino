#define FORWARD 0x09
#define BACKWARD 0x06
#define calc_PWM(_per)((unsigned int)(_per*2.55))
#define STOP 0x00

int Motor[6] = {22,23,24,25,4,5};
void setup() {
  int z;
  for(z=0;z<6;z++)
  {
    pinMode(Motor[z],OUTPUT);
    digitalWrite(Motor[z],LOW);
  }
  Motor_Model(FORWARD);

  Serial.begin(38400);
  Serial2.begin(115200);
}
unsigned char recv[7]={0};
unsigned char tmp_recv=0;
unsigned char last_recv=0;
int count=0;
 long int sp,angle;
 char sned[10];
 int no_data=0;
void loop() {

  
//  if(no_data>0)
//  {
//      no_data--;
//      delay(1);
//  }
//  else
//  {
//       speed(0,0);
//  }
  if(Serial.available()>0)
  {
    //Serial2.write(Serial.read());
      tmp_recv=Serial.read();                

      if( tmp_recv==0xAA)
      {
          memset(recv,0,7);
          count=1;
          goto end;
      }

      if(count>0)
      {
          recv[count++]=tmp_recv;
      }
      if(count==6)
      {
              no_data=1000;
              count=0;
              sp = (unsigned int)recv[1] + (unsigned int)recv[2]*255;
              angle = (unsigned int)recv[3] + (unsigned int)recv[4]*255;
//              if(angle>2000)
//              {
//                  angle=2000; 
//              }
//              if(angle<1000)
//              {
//                  angle=1000;  
//              }
              if(sp>1600)
              {
                sp=1600; 
               }
               if(sp<1400)
               {
                  sp=1400;
                }
              angle = angle-1500;
              sp = sp - 1500;
              if(sp==1500)
              {
                 speed(0,0);
              }
              else
              {
                  speed(sp-(angle)*0.2,sp+(angle)*0.2);
              }
      }
      end:
        last_recv=tmp_recv;

  }

 
}
void speed(int L,int R)
{
  unsigned int OC_value = 0;
    if(L>100)
    {
        L=100;
    }
    else if(L<-100)
    {
      L=-100; 
    }
    if(R>100)
    {
      R=100;
    }
    else if(R<-100)
    {
      R=-100;  
    }
  
    
    if(L>=0)
    {    
      digitalWrite(Motor[0],HIGH);  
            digitalWrite(Motor[1],LOW);  

    }  
    else
    {
         digitalWrite(Motor[0],LOW);  
            digitalWrite(Motor[1],HIGH);  
    }
    if(R>=0)
    {    
      digitalWrite(Motor[2],LOW);  
            digitalWrite(Motor[3],HIGH);  

    }  
    else
    {
         digitalWrite(Motor[2],HIGH);  
            digitalWrite(Motor[3],LOW);  
    }

    L = abs(L);
    R = abs(R);
    analogWrite(Motor[4],calc_PWM(L));   
    analogWrite(Motor[5],calc_PWM(R));
}

void Motor_Model(int da)
{
  int z;
  for(z=0;z<4;z++)
  {
      digitalWrite(Motor[z],(da>>z)&0x01);
  }
}
