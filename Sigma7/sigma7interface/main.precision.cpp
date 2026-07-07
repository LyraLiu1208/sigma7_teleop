/*******************************************************************************
Sigma.7 precision probe
*******************************************************************************
Separate from the original main.cpp.
This probe prints position with higher precision, raw encoder counts, and
device status so we can rule out display rounding as the reason values look
constant.
*******************************************************************************/

#include <stdio.h>
#include <string.h>
#include "dhdc.h"

#define REFRESH_INTERVAL 0.1

static void printStatusBits (const int status[DHD_MAX_STATUS])
{
    printf("status[power=%d connected=%d started=%d reset=%d idle=%d force=%d brake=%d torque=%d wrist=%d error=%d gravity=%d timeguard=%d wrist_init=%d redundancy=%d forceoff=%d locks=%d]\n",
           status[DHD_STATUS_POWER],
           status[DHD_STATUS_CONNECTED],
           status[DHD_STATUS_STARTED],
           status[DHD_STATUS_RESET],
           status[DHD_STATUS_IDLE],
           status[DHD_STATUS_FORCE],
           status[DHD_STATUS_BRAKE],
           status[DHD_STATUS_TORQUE],
           status[DHD_STATUS_WRIST_DETECTED],
           status[DHD_STATUS_ERROR],
           status[DHD_STATUS_GRAVITY],
           status[DHD_STATUS_TIMEGUARD],
           status[DHD_STATUS_WRIST_INIT],
           status[DHD_STATUS_REDUNDANCY],
           status[DHD_STATUS_FORCEOFFCAUSE],
           status[DHD_STATUS_LOCKS]);
}

int main ()
{
    double px = 0.0, py = 0.0, pz = 0.0;
    double qx = 0.0, qy = 0.0, qz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double wx = 0.0, wy = 0.0, wz = 0.0;
    double pg = 0.0, vg = 0.0, fg = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double tx = 0.0, ty = 0.0, tz = 0.0;
    int d0 = 0, d1 = 0, d2 = 0;
    int w0 = 0, w1 = 0, w2 = 0;
    int ge = 0;
    int status[DHD_MAX_STATUS];
    double last_px = 0.0, last_py = 0.0, last_pz = 0.0;
    int last_d0 = 0, last_d1 = 0, last_d2 = 0;
    double t0 = dhdGetTime ();
    double t1 = t0;
    int done = 0;
    int major = 0, minor = 0, release = 0, revision = 0;

    setvbuf(stdout, NULL, _IONBF, 0);

    dhdGetSDKVersion (&major, &minor, &release, &revision);
    printf ("\nSigma7 precision probe\n");
    printf ("SDK %d.%d.%d.%d\n", major, minor, release, revision);

    printf ("device count=%d available=%d\n", dhdGetDeviceCount(), dhdGetAvailableCount());

    if (dhdOpen() < 0)
    {
        printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
        return -1;
    }

    printf ("%s device detected\n", dhdGetSystemName());
    printf ("device id=%d\n", dhdGetDeviceID());
    printf ("press 'q' to quit\n\n");

    dhdEnableForce (DHD_ON);

    while (!done)
    {
        memset(status, 0, sizeof(status));
        dhdGetStatus(status);

        dhdUpdateEncoders();
        int r_pos = dhdGetPosition(&px, &py, &pz);
        int r_ori = dhdGetOrientationRad(&qx, &qy, &qz);
        int r_grip = dhdGetGripperAngleRad(&pg);
        int r_lin = dhdGetLinearVelocity (&vx, &vy, &vz);
        int r_ang = dhdGetAngularVelocityRad (&wx, &wy, &wz);
        int r_gvel = dhdGetGripperLinearVelocity (&vg);
        int r_force = dhdGetForceAndTorqueAndGripperForce(&fx, &fy, &fz, &tx, &ty, &tz, &fg);

        dhdGetDeltaEncoders(&d0, &d1, &d2);
        dhdGetWristEncoders(&w0, &w1, &w2);
        dhdGetGripperEncoder(&ge);

        if (r_pos < DHD_NO_ERROR || r_ori < DHD_NO_ERROR || r_grip < DHD_NO_ERROR ||
            r_lin < DHD_NO_ERROR || r_ang < DHD_NO_ERROR || r_gvel < DHD_NO_ERROR)
        {
            printf ("error: DHD read failed (%s)\n", dhdErrorGetLastStr());
            printf ("  pos=%d ori=%d grip=%d lin=%d ang=%d gvel=%d\n",
                    r_pos, r_ori, r_grip, r_lin, r_ang, r_gvel);
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
            printf ("p=(%+.6f %+.6f %+.6f) ", px, py, pz);
            printf ("dp=(%+.6f %+.6f %+.6f) ", px - last_px, py - last_py, pz - last_pz);
            printf ("q=(%+.6f %+.6f %+.6f) ", qx, qy, qz);
            printf ("g=(%+.6f) ", pg);
            printf ("v=(%+.6f %+.6f %+.6f) ", vx, vy, vz);
            printf ("w=(%+.6f %+.6f %+.6f) ", wx, wy, wz);
            printf ("enc[delta=%d,%d,%d wrist=%d,%d,%d grip=%d] ", d0, d1, d2, w0, w1, w2, ge);
            printf ("denc=%d,%d,%d ", d0 - last_d0, d1 - last_d1, d2 - last_d2);
            printf ("rc[pos=%d ori=%d grip=%d lin=%d ang=%d gvel=%d force=%d] ", r_pos, r_ori, r_grip, r_lin, r_ang, r_gvel, r_force);
            printStatusBits(status);
            printf ("\r");

            last_px = px;
            last_py = py;
            last_pz = pz;
            last_d0 = d0;
            last_d1 = d1;
            last_d2 = d2;
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
