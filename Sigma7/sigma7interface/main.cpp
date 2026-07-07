/*******************************************************************************
Sigma.7 interface for impedance control
Luka Peternel
l.peternel@tudelft.nl
*******************************************************************************/



#include <stdio.h>
#include <math.h>
#include "dhdc.h"

#define REFRESH_INTERVAL        0.1   //  seconds
#define LINEAR_STIFFNESS      100.0   //  N/m
#define ANGULAR_STIFFNESS      5.0   //  Nm/rad
#define LINEAR_VISCOSITY       20.0   //  N/(m/s)
#define ANGULAR_VISCOSITY       0.03  // Nm/(rad/s)



int
main (int  argc,
      char **argv)
{
    double px, py, pz;
    double qx, qy, qz;
    double vx, vy, vz;
    double wx, wy, wz;
    double pg, vg, fg;
    double fx, fy, fz;
    double tx, ty, tz;
    double t1,t0  = dhdGetTime ();
    int    done   = 0;

    double pdx, pdy, pdz, qdx, qdy, qdz, pdg;
    double fdx, fdy, fdz;
    double tdx, tdy, tdz;

    pdx = -0.035;
    pdy = +0.035;
    pdz = +0.000;
    qdx = +0.0;
    qdy = +0.0;
    qdz = +0.0;
    pdg = +0.510;

    // message
    int major, minor, release, revision;
    dhdGetSDKVersion (&major, &minor, &release, &revision);
    printf ("\n");
    printf ("Impedance control");
    printf ("\n\n");

    // open the first available device
    if (dhdOpen() < 0)
    {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
    dhdSleep (2.0);
    return -1;
    }

    // identify device
    printf ("%s device detected\n\n", dhdGetSystemName());

    // display instructions
    printf ("press 'q' to quit\n\n");

    // enable force
    dhdEnableForce (DHD_ON);

    // haptic loop
    while (!done)
    {
        dhdGetPosition(&px, &py, &pz);
        dhdGetOrientationRad(&qx, &qy, &qz);
        dhdGetGripperAngleRad(&pg);
        dhdGetLinearVelocity (&vx, &vy, &vz);
        dhdGetAngularVelocityRad (&wx, &wy, &wz);
        dhdGetGripperLinearVelocity (&vg);
//        dhdGetForce(&fx, &fy, &fz);
        dhdGetForceAndTorqueAndGripperForce(&fx, &fy, &fz, &tx, &ty, &tz, &fg);

        fdx = LINEAR_STIFFNESS * (pdx-px) -2 * 0.7 * sqrt(LINEAR_STIFFNESS) * vx;
        fdy = LINEAR_STIFFNESS * (pdy-py) -2 * 0.7 * sqrt(LINEAR_STIFFNESS) * vy;
        fdz = LINEAR_STIFFNESS * (pdz-pz) -2 * 0.7 * sqrt(LINEAR_STIFFNESS) * vz;

        tdx = ANGULAR_STIFFNESS * (qdx-qx) -ANGULAR_VISCOSITY * wx;
        tdy = -ANGULAR_VISCOSITY * wy;
        tdz = -ANGULAR_VISCOSITY * wz;

        //tdx = ANGULAR_STIFFNESS * (qdx-qx) -ANGULAR_VISCOSITY * wx;
        //tdy = ANGULAR_STIFFNESS * (qdx-qx) -ANGULAR_VISCOSITY * wy;
        //tdz = ANGULAR_STIFFNESS * (qdx-qx) -ANGULAR_VISCOSITY * wz;

        fg = ANGULAR_STIFFNESS * (pdg-pg) -LINEAR_VISCOSITY * vg;

        // apply force
        if (dhdSetForceAndTorqueAndGripperForce (fdx, fdy, fdz, tdx, tdy, tdz, fg) < DHD_NO_ERROR)
        {
          printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
          done = 1;
        }

        // display refresh rate and position at 10Hz
        t1 = dhdGetTime ();
        if ((t1-t0) > REFRESH_INTERVAL)
        {

            // update timestamp
            t0 = t1;

            printf ("p (%+0.03f %+0.03f %+0.03f %+0.03f) m  ", px, py, pz, pg);
            printf ("q (%+0.03f %+0.03f %+0.03f) rad  ", qx, qy, qz);

            // write down velocity
            //printf ("v (%+0.03f %+0.03f %+0.03f) m/s  ", vx, vy, vz);
            //if (dhdHasWrist   ()) printf ("|  w (%+02.01f %+02.01f %+02.01f) rad/s  ", wx, wy, wz);
            //if (dhdHasGripper ()) printf ("|  vg (%+0.03f) m/s  ", vg);

            printf ("f (%+0.03f %+0.03f %+0.03f %+0.03f) N  ", fx, fy, fz, fg);

            printf ("\r");

            // user input
            if (dhdKbHit() && dhdKbGet() == 'q') done = 1;
        }
    }

    // close the connection
    dhdClose ();

    // happily exit
    printf ("\ndone.\n");
    return 0;
}
