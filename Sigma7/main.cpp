/*******************************************************************************
Sigma.7 interface code for teleimpedance

Luka Peternel
l.peternel@tudelft.nl
*******************************************************************************/



#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdexcept>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <termios.h>
//#include <stropts.h>
#include <sys/ioctl.h>

#include <vector>
#include <sstream>
#include <iostream>

#include "dhdc.h"

#define TIME_MONITOR            0.1     //  seconds
#define TIME_STEP               0.006   //  seconds
#define LINEAR_STIFFNESS      100.0     //  N/m
#define ANGULAR_STIFFNESS      10.0     //  Nm/rad
#define LINEAR_VISCOSITY       20.0     //  N/(m/s)
#define ANGULAR_VISCOSITY       0.03    // Nm/(rad/s)



int main ()
{
    double px, py, pz;
    double vx, vy, vz;
    double t1, t0, t3, t2  = dhdGetTime ();
    int done = 0;
    int setzero = 0;
    int sending = 0;
    char keyinput = '0';
    double offset[3] = {0,0,0};

    std::vector<int> vect;
    vect.assign(9,0);




    // recv UDP
    #define BUFLEN1 37//29  // maximum length of buffer

    struct sockaddr_in si_me1, si_other1;
    socklen_t slen1 = sizeof(si_other1);

    int s1, recv_len1;
    char recvData[BUFLEN1];

    // create a UDP socket
    if ((s1=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
        printf("socket creation failed");
    }

    // zero out the structure
    memset((char *) &si_me1, 0, sizeof(si_me1));

    si_me1.sin_family = AF_INET;
    si_me1.sin_port = htons(41001);
    si_me1.sin_addr.s_addr = htonl(INADDR_ANY);

    // bind socket to port
    if( bind(s1 , (struct sockaddr*)&si_me1, sizeof(si_me1) ) == -1)
    {
        printf("socket binding failed");
    }



    // send UDP
    #define BUFLEN3 104  // maximum length of buffer

    struct sockaddr_in si_other3;

    int s3;
    double sendData[13];
    sendData[0] = 0; // receive flag for remote robot

    // create a UDP socket
    if ((s3=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
        printf("socket creation failed");
    }

    memset(&si_other3,0,sizeof(si_other3));
    si_other3.sin_family = AF_INET;
    si_other3.sin_port = htons(42000);
    si_other3.sin_addr.s_addr = inet_addr("127.0.0.1");
    bzero(&(si_other3.sin_zero),8);

    setvbuf(stdout, NULL, _IONBF, 0);







    // instructions
    printf ("\n");
    printf ("-------------------------------------------------------------------------\n");
    printf ("TELEIMPEDANCE INTERFACE\n");
    printf ("\n");
    printf ("Options:\n");
    printf ("press 'q' to quit\n");
    printf ("press 'r' to set offset position\n");
    printf ("press 'o' open connection to remote robot\n");
    printf ("press 'p' pause connection to remote robot\n");
    printf ("-------------------------------------------------------------------------\n\n");

    // open available device
    if (dhdOpen() < 0)
    {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
    dhdSleep (2.0);
    return -1;
    }

    // identify device
    printf ("%s device detected\n", dhdGetSystemName());

    printf ("\nProgram started.\n\n\n");

    // enable force
    dhdEnableForce (DHD_ON);

    // main loop
    while (!done)
    {

        // keep it real time with the desired time step
        t1 = dhdGetTime ();
        if ((t1-t0) > TIME_STEP)
        {
            t0 = t1; // update timestamp

            // get data from ellipsoid interface
            recv_len1 = recvfrom(s1, recvData, BUFLEN1, MSG_DONTWAIT, (struct sockaddr *) &si_other1, &slen1);

            if (recv_len1 > 0)
            {
                std::string receivedString;
                receivedString.assign(&(recvData[0]),BUFLEN1);
                std::stringstream ss(receivedString);
                vect.clear();

                for (int i; ss >> i;)
                {
                    vect.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
            }

            // get poistion from haptic interface
            dhdGetPosition(&px, &py, &pz);
            dhdGetLinearVelocity (&vx, &vy, &vz);

//            sendData[1] = -(py - offset[1]);
//            sendData[2] = px - offset[0];
            sendData[1] = px - offset[0];
            sendData[2] = py - offset[1];
            sendData[3] = pz - offset[2];

            for ( int i = 0; i < 9; i++)
                sendData[4+i] = static_cast<double>(vect[i]);

//            double tmp = sendData[4];
//            sendData[4] = sendData[8];
//            sendData[8] = tmp;

            // send commands to remote robot
            if(sending == 1)
                sendto(s3, (char*) sendData, BUFLEN3, 0, (struct sockaddr *)&si_other3, sizeof(si_other3));


            // apply force to haptic interface
            if (dhdSetForceAndTorqueAndGripperForce (0, 0, 0, 0, 0, 0, 0) < DHD_NO_ERROR)
            {
              printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
              done = 1;
            }

        }




        // monitoring with monitoring time step
        t3 = dhdGetTime ();
        if ((t3-t2) > TIME_MONITOR)
        {
            t2 = t3; // update timestamp
            printf ("p=(%+0.03f %+0.03f %+0.03f)m   ", px, py, pz);
            //printf ("v=(%+0.03f %+0.03f %+0.03f)m/s   ", vx, vy, vz);
            printf ("o=(%+0.03f %+0.03f %+0.03f)m   ", offset[0], offset[1], offset[2]);
            printf ("e=(%0.01f %0.01f %0.01f)   ", sendData[4], sendData[5], sendData[6]);
            printf ("e=(%0.01f %0.01f %0.01f)   ", sendData[7], sendData[8], sendData[9]);
            printf ("e=(%0.01f %0.01f %0.01f)   ", sendData[10], sendData[11], sendData[12]);
            printf ("send=%i   ", sending);
            printf ("\r");

            // check keyboard input
            if (dhdKbHit()) keyinput = dhdKbGet();
            if (keyinput == 'q') done = 1;
            if (keyinput == 'r') setzero = 1;
            if (keyinput == 'o')
            {
                sending = 1;
                sendData[0] = 3;
            }
            if (keyinput == 'p')
            {
                sending = 0;
                sendData[0] = 3;
            }
            keyinput = '0';


            // set zero position for teleoperation
            if (setzero == 1)
            {
                offset[0] = px;
                offset[1] = py;
                offset[2] = pz;
                setzero = 0;
            }
        }


    }

    // close the connection
    dhdClose ();

    printf ("\nProgram terminated.\n");
    return 0;
}
