/*******************************************************************************
Sigma.7 interface debug probe
*******************************************************************************
This file is separate from main.cpp.
It prints DHD return codes and the live pose values so we can verify whether
the device readings actually change.
*******************************************************************************/

#include <stdio.h>
#include <math.h>
#include "dhdc.h"

#define REFRESH_INTERVAL 0.1

int main ()
{
    double px = 0.0, py = 0.0, pz = 0.0;
    double qx = 0.0, qy = 0.0, qz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double wx = 0.0, wy = 0.0, wz = 0.0;
    double pg = 0.0, vg = 0.0, fg = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double tx = 0.0, ty = 0.0, tz = 0.0;
    double t0 = dhdGetTime ();
    double t1 = t0;
    int done = 0;
    int major = 0, minor = 0, release = 0, revision = 0;

    setvbuf(stdout, NULL, _IONBF, 0);

    dhdGetSDKVersion (&major, &minor, &release, &revision);
    printf ("\nSigma7 debug probe\n");
    printf ("SDK %d.%d.%d.%d\n\n", major, minor, release, revision);

    if (dhdOpen() < 0)
    {
        printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
        return -1;
    }

    printf ("%s device detected\n\n", dhdGetSystemName());
    printf ("press 'q' to quit\n\n");

    dhdEnableForce (DHD_ON);

    while (!done)
    {
        int r_pos = dhdGetPosition(&px, &py, &pz);
        int r_ori = dhdGetOrientationRad(&qx, &qy, &qz);
        int r_grip = dhdGetGripperAngleRad(&pg);
        int r_lin = dhdGetLinearVelocity (&vx, &vy, &vz);
        int r_ang = dhdGetAngularVelocityRad (&wx, &wy, &wz);
        int r_gvel = dhdGetGripperLinearVelocity (&vg);
        int r_force = dhdGetForceAndTorqueAndGripperForce(&fx, &fy, &fz, &tx, &ty, &tz, &fg);

        if (r_pos < DHD_NO_ERROR || r_ori < DHD_NO_ERROR || r_grip < DHD_NO_ERROR ||
            r_lin < DHD_NO_ERROR || r_ang < DHD_NO_ERROR || r_gvel < DHD_NO_ERROR ||
            r_force < DHD_NO_ERROR)
        {
            printf ("error: DHD read failed\n");
            printf ("  pos=%d ori=%d grip=%d lin=%d ang=%d gvel=%d force=%d\n",
                    r_pos, r_ori, r_grip, r_lin, r_ang, r_gvel, r_force);
            printf ("  last=%s\n", dhdErrorGetLastStr());
            break;
        }

        if (dhdSetForceAndTorqueAndGripperForce(0, 0, 0, 0, 0, 0, 0) < DHD_NO_ERROR)
        {
            printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
            break;
        }

        t1 = dhdGetTime ();
        if ((t1 - t0) > REFRESH_INTERVAL)
        {
            t0 = t1;
            printf ("p=(%+0.03f %+0.03f %+0.03f) ", px, py, pz);
            printf ("q=(%+0.03f %+0.03f %+0.03f) ", qx, qy, qz);
            printf ("g=(%+0.03f) ", pg);
            printf ("v=(%+0.03f %+0.03f %+0.03f) ", vx, vy, vz);
            printf ("w=(%+0.03f %+0.03f %+0.03f) ", wx, wy, wz);
            printf ("fg=(%+0.03f) ", fg);
            printf ("rc[pos=%d ori=%d grip=%d lin=%d ang=%d gvel=%d force=%d]  ",
                    r_pos, r_ori, r_grip, r_lin, r_ang, r_gvel, r_force);
            printf ("\r");
        }

        if (dhdKbHit() && dhdKbGet() == 'q')
        {
            done = 1;
        }
    }

    dhdClose ();
    printf ("\ndone.\n");
    return 0;
}
