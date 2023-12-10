import numpy as np
import sys



class DPsnake():

    def __init__(self, init_point:np.ndarray, gradient:np.ndarray, alpha:float=1.0) -> None:
        
        self.point = init_point.astype(np.int64) + 1
        self.mag_g = np.pad(
            gradient**2, ((1,1),(1,1)), 
            mode='constant',constant_values=(-np.inf,-np.inf)
        )
        self.shift_array = np.array([[-1,0], [1,0], [0,-1], [0, 1]])
        self.DIRMAP = ["up", "down", "left", "right"]
    
        self.a = alpha
        self.transition = np.zeros((4, self.point.shape[0]))
    
    def neighbor_coo(self, v:int)->np.ndarray:
        
        coor = [[self.point[v][0], self.point[v][1]] for _ in range(4)]
        
        for i in range(4):
            coor[i][i//2] += (-1)**((i+1)%4)
        
        return np.array(coor, dtype=np.int32)

    def distance(self, vi:np.ndarray, vj:np.ndarray)->np.ndarray:
        #vi :dst, vj :src
        def is_vaild_pair(x:np.ndarray, y:np.ndarray)->bool:
            for i in range(2):
                if x[i] > self.mag_g.shape[i] -2 :
                    return False
                if y[i] > self.mag_g.shape[i] -2 :
                    return False
            return True
        
        d = np.zeros((vj.shape[0], vi.shape[0]))
        for dst in range(vj.shape[0]):
            for src in range(vi.shape[0]):
                if is_vaild_pair(vi[dst], vj[src]):
                    d[dst][src] = np.sum((vi[dst]-vj[src])**2)
                else:
                    d[dst][src] = np.inf

        return d

    def energy(self, distance:np.ndarray, g_slope:np.ndarray)->np.ndarray:
        return -g_slope + self.a*distance

    def __print_log(
            self, vi:int, 
            neighbor_coos:np.ndarray, g_slope:np.ndarray, 
            d:np.ndarray, dst_e:np.ndarray, e:np.ndarray
        )->None:

        srccoor = self.point[vi-1]
        dstcoor = self.point[vi]
        print(f"+++++++ src {vi-1} {srccoor}", end=" ")
        print(f"dst node {vi} {dstcoor} +++++++")
        print(f"====== {srccoor} neighbor ========")
        print(neighbor_coos[vi-1])
        print(f"====== {dstcoor} neighbor ========")
        print(neighbor_coos[vi])
        print(f"====== {srccoor} gradient^2 ========")
        print(g_slope[vi-1].reshape(1, -1))
        print(f"====== dst_{vi} x src_{vi-1} pairwise distance ========")
        print(d)
        print(f"====== dst_{vi} enery ========")
        print(dst_e)
        print(f"======  dst_{vi} x src_{vi-1} pairwise energy ========")
        print(e)
        print(f"========== dp table =========")
        print(self.transition)
        print("++++++++++++++++++++++++++++++++++++++++++++")

    def __call__(self, T:int, log: bool=False)->np.ndarray:

        self.transition = self.transition*0

        for _ in range(T):
            
            neighbor_coos = [ self.point[vi] + self.shift_array for vi in range(self.point.shape[0])]
            g_slope = [self.mag_g[n[:, 0], n[:, 1]] for n in neighbor_coos]
            self.transition[:, 0] = -g_slope[0]
            if log:
                print("++++++++++++++++++ initial +++++++++++++++++++++")
                print(self.transition)
                print('++++++++++++++++++++++++++++++++++++++++++++++++')
            
            dst_behavior = np.arange(4)
            track = np.zeros((self.point.shape[0]-1 ,4), dtype=np.int32)
                        
            for vi in range(1, self.point.shape[0]):
                
                src_e = self.transition[:, vi-1]
            
                d = self.distance(neighbor_coos[vi],neighbor_coos[vi-1])
                dst_e = self.energy(d, g_slope[vi-1].reshape(1, -1))
            
                e = src_e.reshape(1, -1)+dst_e
              
                min_src_behavior = np.argmin(e, axis=1)

                self.transition[:, vi] = e[dst_behavior,min_src_behavior]
                track[vi-1] = min_src_behavior
                
                if log:
                    self.__print_log(vi, neighbor_coos, g_slope, d, dst_e, e)
 
            if log:
                print(f"----------------------- iteraion {_} over----------------------")
        
            self.__backtrack(track=track, log=log)
            
            
    def __backtrack(self,track:np.ndarray, log:bool=False)->None:
        
        src_dir = np.argmin(self.transition[:, -1])
        if log:
            print("============backtrack===================")
            print(f"last node shift direction : {self.DIRMAP[src_dir]}")
            print(track)

        for vi in range(self.point.shape[0] - 1, -1, -1):
            print(f"src nodel {vi} {self.point[vi]}, shift dir : {self.DIRMAP[src_dir]}")
            self.point[vi] += self.shift_array[src_dir]
            if vi > 0:
                src_dir = track[vi-1][src_dir]
        
        if log:
            print("=============================================")
    
    def get_enery_table(self)->np.ndarray:
        return self.transition.copy()
    
    def get_points(self)->np.ndarray:
        return self.point-1


def main():

    init = np.loadtxt("init_snake.csv", delimiter=",")
    img_gradient = np.loadtxt("gradient.csv", delimiter=",") 
    snake = DPsnake(init_point=init, gradient=img_gradient)
    max_iter = int(sys.argv[1])
    snake(T=max_iter, log=True)
    
    print(f"After {max_iter} iteration : ")
    print(snake.get_enery_table())
    print(snake.get_points()+1)

if __name__ == "__main__":
    main()
            
