from transforms3d.euler import euler2quat, quat2euler
import numpy as np

def get_euler(quat):
    angles = quat2euler(quat)
    angles_deg = [angle*180/np.pi for angle in angles]
    return angles_deg

def get_quat(euler):
    euler_rad = [euler_angle*np.pi/180 for euler_angle in euler]
    quat = euler2quat(*euler_rad)
    quat = [q for q in quat]
    return quat

if __name__ == "__main__":
    euler_in = [-1.75, 43.5, 23] # RPY in degrees
    quat_out = get_quat(euler=euler_in)
    print("The quaternion angles is ", quat_out)

    quat_in = [ 0.90857032, -0.0746846 ,  0.3670863 ,  0.18485085]
    euler_out = get_euler(quat_in)
    print("The Euler angles in degrees is ", euler_out)
    