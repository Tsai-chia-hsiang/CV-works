import os
import os.path as osp
import numpy as np
from numpy import linalg as linalg

class World3D_Camera2D_Projector():
    
    def __init__(self, world3d:np.ndarray, camera2d:np.ndarray) -> None:
        
        self.__world3d = World3D_Camera2D_Projector.translate(world3d)
        self.__camera2d = World3D_Camera2D_Projector.translate(camera2d)
        self.__projection_3d2d:np.ndarray = self.__estimate_projection()
        _, self.native_mean_err = self.predict(world3d_coo=self.__world3d, gt_carte=camera2d)
        self.__component:dict = None
    
    @staticmethod
    def translate(x:np.ndarray, flag="CARTE2HOMO")->np.ndarray:
        if flag == "HOMO2CARTE":
            d = x.shape[1] -1 
            f = np.repeat(x[:,-1].reshape(-1, 1), repeats=d, axis=1)
            return x[:, :-1]/f
    
        pendding_shape = (x.shape[0],1)
        return np.hstack((x, np.ones(pendding_shape, dtype=x.dtype)))
    
    def __estimate_projection(self)->np.ndarray:
        """
        using minimum eigen value for A^TA
        to apporximate
        """
        r2 = np.repeat(self.__world3d, repeats=2 ,axis=0)
        n = self.__camera2d.shape[0]
        def leftcols():
            left0 = r2.copy()
            left1 = r2.copy()
            left0[1::2, :] = 0
            left1[::2, :] = 0
            return np.hstack((left0,left1))
    
        def rightcols():
            cam_x = np.repeat(self.__camera2d[:, :-1].reshape(2*n, 1), repeats=4, axis=1)
            return r2*(-cam_x)
        
        A =  np.hstack((leftcols(), rightcols()))
        e, v = linalg.eig(A.T@A)
        return v[:,np.argmin(e)].reshape((3,4))

    def predict(self, world3d_coo:np.ndarray, gt_carte:np.ndarray)->np.ndarray:
        
        pro = self.project(world3d_coo=world3d_coo)
        meanerr = np.mean( np.sum((gt_carte-pro)**2, axis=1)**0.5 )
        return pro, meanerr

    def project(self, world3d_coo:np.ndarray)->np.ndarray:
        
        w = world3d_coo if world3d_coo.shape[1] == 4 else World3D_Camera2D_Projector.translate(world3d_coo)
        pred_homo =  (self.__projection_3d2d@w.T).T
        return World3D_Camera2D_Projector.translate(pred_homo,flag="HOMO2CARTE")

    def decompose_projection_matrix(self)->dict:
        
        """
        P = C[R|T] = CR+CT
        CR = P[:3, :]; CT = P[3, :]
        """
        if self.__component is not None:
            return self.__component.copy()
        #Q, R = P33^(-1)
        rotation_inv, calibration_inv = linalg.qr(linalg.inv(self.__projection_3d2d[:, :3]))
        self.__component =  {
            'calibration':linalg.inv(calibration_inv),  
            'rotation':rotation_inv.T, 
            'translation':calibration_inv@(self.__projection_3d2d[:,3:4])
        }
        return self.__component.copy()
           
    def P(self):
        return self.__projection_3d2d.copy()

def main():

    if not osp.exists(osp.join('projection')):
        os.mkdir(osp.join('projection'))
    
    realworld = np.loadtxt(osp.join("sample","3d.csv"),delimiter=',') # 3D point pairs
    camera = np.loadtxt(osp.join("sample","2d.csv"), delimiter=',') # 2D camera point pairs

    # calculate projection matrix
    projtor = World3D_Camera2D_Projector(world3d=realworld, camera2d=camera) 
    print(f"Average Projection Error: {projtor.native_mean_err}")
    np.savetxt(osp.join("projection","p.csv"), projtor.P(), delimiter=',')

    # decompose projection matrix to get 
    # calibration  matrix rotation matrix and translation matrix
    components = projtor.decompose_projection_matrix()
    for component, matrix in components.items():
        np.savetxt(osp.join("projection",f"{component}.csv"), matrix, delimiter=',')


if __name__ == "__main__":
    main()
