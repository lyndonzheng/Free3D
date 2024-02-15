import torch
import numpy as np
from scipy.spatial.transform import Rotation


def cartesian_to_spherical(R, T):
    '''
    transfer the camera rotation and translation to the spherical cooridnate system
    R: rotation with [B, 3, 3]
    T: translation with [B, 3]
    '''
    xyz = - torch.einsum('bij,bi->bj', R, T)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = torch.sqrt(xy + xyz[:,2]**2)
    theta = torch.arctan2(torch.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    azimuth = torch.arctan2(xyz[:,1], xyz[:,0])
    return theta.unsqueeze(1), azimuth.unsqueeze(1), z.unsqueeze(1)

class Pose():
    """
    A class of operations on camera poses (PyTorch tensors with shape [...,3,4])
    each [3,4] camera pose takes the form of [R|t]
    """

    def __call__(self,R=None,t=None):
        # construct a camera pose from the given R and/or t
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        # invert a camera pose
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # compose a sequence of poses together
        # pose_new(x) = poseN o ... o pose2 o pose1(x)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b o pose_a(x)
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

class Lie():
    """
    Lie algebra for SO(3) and SE(3) operations in PyTorch
    """

    def so3_to_SO3(self,w): # [...,3]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def SO3_to_so3(self,R,eps=1e-7): # [...,3,3]
        trace = R[...,0,0]+R[...,1,1]+R[...,2,2]
        theta = ((trace-1)/2).clamp(-1+eps,1-eps).acos_()[...,None,None]%np.pi # ln(R) will explode if theta==pi
        lnR = 1/(2*self.taylor_A(theta)+1e-8)*(R-R.transpose(-2,-1)) # FIXME: wei-chiu finds it weird
        w0,w1,w2 = lnR[...,2,1],lnR[...,0,2],lnR[...,1,0]
        w = torch.stack([w0,w1,w2],dim=-1)
        return w

    def se3_to_SE3(self,wu): # [...,3]
        w,u = wu.split([3,3],dim=-1)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        C = self.taylor_C(theta)
        R = I+A*wx+B*wx@wx
        V = I+B*wx+C*wx@wx
        Rt = torch.cat([R,(V@u[...,None])],dim=-1)
        return Rt

    def SE3_to_se3(self,Rt,eps=1e-8): # [...,3,4]
        R,t = Rt.split([3,1],dim=-1)
        w = self.SO3_to_so3(R)
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[...,None,None]
        I = torch.eye(3,device=w.device,dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        u = (invV@t)[...,0]
        wu = torch.cat([w,u],dim=-1)
        return wu    

    def skew_symmetric(self,w):
        w0,w1,w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                          torch.stack([w2,O,-w0],dim=-1),
                          torch.stack([-w1,w0,O],dim=-1)],dim=-2)
        return wx

    def taylor_A(self,x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_B(self,x,nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    def taylor_C(self,x,nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

class Quaternion():

    def q_to_R(self,q):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        qa,qb,qc,qd = q.unbind(dim=-1)
        R = torch.stack([torch.stack([1-2*(qc**2+qd**2),2*(qb*qc-qa*qd),2*(qa*qc+qb*qd)],dim=-1),
                         torch.stack([2*(qb*qc+qa*qd),1-2*(qb**2+qd**2),2*(qc*qd-qa*qb)],dim=-1),
                         torch.stack([2*(qb*qd-qa*qc),2*(qa*qb+qc*qd),1-2*(qb**2+qc**2)],dim=-1)],dim=-2)
        return R

    def R_to_q(self,R,eps=1e-8): # [B,3,3]
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # FIXME: this function seems a bit problematic, need to double-check
        row0,row1,row2 = R.unbind(dim=-2)
        R00,R01,R02 = row0.unbind(dim=-1)
        R10,R11,R12 = row1.unbind(dim=-1)
        R20,R21,R22 = row2.unbind(dim=-1)
        t = R[...,0,0]+R[...,1,1]+R[...,2,2]
        r = (1+t+eps).sqrt()
        qa = 0.5*r
        qb = (R21-R12).sign()*0.5*(1+R00-R11-R22+eps).sqrt()
        qc = (R02-R20).sign()*0.5*(1-R00+R11-R22+eps).sqrt()
        qd = (R10-R01).sign()*0.5*(1-R00-R11+R22+eps).sqrt()
        q = torch.stack([qa,qb,qc,qd],dim=-1)
        for i,qi in enumerate(q):
            if torch.isnan(qi).any():
                K = torch.stack([torch.stack([R00-R11-R22,R10+R01,R20+R02,R12-R21],dim=-1),
                                 torch.stack([R10+R01,R11-R00-R22,R21+R12,R20-R02],dim=-1),
                                 torch.stack([R20+R02,R21+R12,R22-R00-R11,R01-R10],dim=-1),
                                 torch.stack([R12-R21,R20-R02,R01-R10,R00+R11+R22],dim=-1)],dim=-2)/3.0
                K = K[i]
                eigval,eigvec = torch.linalg.eigh(K)
                V = eigvec[:,eigval.argmax()]
                q[i] = torch.stack([V[3],V[0],V[1],V[2]])
        return q

    def invert(self,q):
        qa,qb,qc,qd = q.unbind(dim=-1)
        norm = q.norm(dim=-1,keepdim=True)
        q_inv = torch.stack([qa,-qb,-qc,-qd],dim=-1)/norm**2
        return q_inv

    def product(self,q1,q2): # [B,4]
        q1a,q1b,q1c,q1d = q1.unbind(dim=-1)
        q2a,q2b,q2c,q2d = q2.unbind(dim=-1)
        hamil_prod = torch.stack([q1a*q2a-q1b*q2b-q1c*q2c-q1d*q2d,
                                  q1a*q2b+q1b*q2a+q1c*q2d-q1d*q2c,
                                  q1a*q2c-q1b*q2d+q1c*q2a+q1d*q2b,
                                  q1a*q2d+q1b*q2c-q1c*q2b+q1d*q2a],dim=-1)
        return hamil_prod

pose = Pose()
lie = Lie()
quaternion = Quaternion()

def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

# basic operations of transforming 3D points between world/camera/image coordinates
def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)


def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)


def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)


def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom@pose_inv.transpose(-1,-2)


def cam2world_legacy(X, pose):
    '''Legacy function to match the original MatchNeRF pretrain_weight; Do NOT use for new experiments.'''
    X_hom = to_hom(X)
    # pose_inv = Pose().invert(pose)
    pose_square = torch.eye(4).unsqueeze(0).repeat(X_hom.shape[0], 1, 1).to(pose.device)
    pose_square[:, :3, :] = pose
    # original code use numpy to get inverse matrix, default in float64
    pose_inv = pose_square.double().inverse()[:, :3, :].to(torch.float32)
    X_world = X_hom@pose_inv.transpose(-1,-2)
    return X_world

def get_center_and_ray(img_h, img_w, pose, intr=None, legacy=False, device='cuda'): # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")

    with torch.no_grad():
        # compute image coordinate grid
        y_range = torch.arange(img_h, dtype=torch.float32,device=device).add_(0. if legacy else 0.5)
        x_range = torch.arange(img_w, dtype=torch.float32,device=device).add_(0. if legacy else 0.5)
        Y,X = torch.meshgrid(y_range, x_range, indexing='ij') # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]

    # compute center and ray
    batch_size = len(pose)
    xy_grid = xy_grid.repeat(batch_size,1,1) # [B,HW,2]
    grid_3D_cam = img2cam(to_hom(xy_grid), intr) 
    center_3D_cam = torch.zeros_like(grid_3D_cam) # [B,HW,3]

    # transform from camera to world coordinates
    c2w_func = cam2world_legacy if legacy else cam2world
    grid_3D = c2w_func(grid_3D_cam, pose) # [B,HW,3]
    center_3D = c2w_func(center_3D_cam, pose) # [B,HW,3]
    ray = grid_3D - center_3D # [B,HW,3]

    return center_3D, ray


def convert_NDC(center,ray,intr,near=1):
    # shift camera center (ray origins) to near plane (z=1)
    # (unlike conventional NDC, we assume the cameras are facing towards the +z direction)
    center = center+(near-center[...,2:])/ray[...,2:]*ray
    # projection
    cx,cy,cz = center.unbind(dim=-1) # [B,HW]
    rx,ry,rz = ray.unbind(dim=-1) # [B,HW]
    scale_x = intr[:,0,0]/intr[:,0,2] # [B]
    scale_y = intr[:,1,1]/intr[:,1,2] # [B]
    cnx = scale_x[:,None]*(cx/cz)
    cny = scale_y[:,None]*(cy/cz)
    cnz = 1-2*near/cz
    rnx = scale_x[:,None]*(rx/rz-cx/cz)
    rny = scale_y[:,None]*(ry/rz-cy/cz)
    rnz = 2*near/cz
    center_ndc = torch.stack([cnx,cny,cnz],dim=-1) # [B,HW,3]
    ray_ndc = torch.stack([rnx,rny,rnz],dim=-1) # [B,HW,3]
    return center_ndc,ray_ndc


def get_interpolate_render_path(c2ws, N_views=30):
    N = len(c2ws)
    rotvec, positions = [], []
    rotvec_inteplat, positions_inteplat = [], []
    weight = np.linspace(1.0, .0, N_views//3, endpoint=False).reshape(-1, 1)
    for i in range(N):
        r = Rotation.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
        positions.append(c2ws[i, :3, 3:].reshape(1, 3))

        if i:
            rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
            positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])

    rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])

    c2ws_render = []
    angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    for rotvec, position in zip(angles_inteplat, positions_inteplat):
        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_euler('xyz', rotvec, degrees=True).as_matrix()
        c2w[:3, 3:] = position.reshape(3, 1)
        c2ws_render.append(c2w.copy())
    c2ws_render = np.stack(c2ws_render)
    return c2ws_render


def spherical_to_camera(theta, azimuth, z):
    # Calculate the camera's position in Cartesian coordinates
    X = z * torch.cos(theta) * torch.cos(azimuth)
    Y = z * torch.cos(theta) * torch.sin(azimuth)
    Z = z * torch.sin(theta)

    camera_T_c2w = torch.cat([X, Y, Z], dim=-1)
    # Calculate the camera's look-at direction vector
    camera_z = - camera_T_c2w
    camera_z /= torch.linalg.norm(camera_z, dim=-1, keepdim=True)
    # Define the camera's up direction vector
    up_direction = torch.zeros((X.size(0), 3)).to(z.device)
    up_direction[:, 2] = - 1.0
    camera_x = torch.cross(up_direction, camera_z, dim=-1)
    camera_x[camera_x.sum(dim=-1) == 0] = torch.tensor([0, 1, 0]).type_as(camera_z)
    camera_x /= torch.linalg.norm(camera_x, dim=-1, keepdim=True)
    camera_y = torch.cross(camera_z, camera_x, dim=-1)
    camera_y =  camera_y / torch.linalg.norm( camera_y, dim=-1, keepdim=True)
    # Construct the translation vector as the negative of the camera position
    C2W = torch.eye(4).unsqueeze(0).repeat(X.size(0),1,1).to(z.device)
    C2W[:, :3, 0] = camera_x
    C2W[:, :3, 1] = camera_y
    C2W[:, :3, 2] = camera_z

    C2W[:, :3, 3] = camera_T_c2w

    # world to cam
    W2C = torch.linalg.inv(C2W)

    return W2C


def transform_points(points, transform, translate=True):
    """ Apply linear transform to a np array of points.
    Args:
        points (np array [..., 3]): Points to transform.
        transform (np array [3, 4] or [4, 4]): Linear map.
        translate (bool): If false, do not apply translation component of transform.
    Returns:
        transformed points (np array [..., 3])
    """
    # Append ones or zeros to get homogenous coordinates
    if translate:
        constant_term = np.ones_like(points[..., :1])
    else:
        constant_term = np.zeros_like(points[..., :1])
    points = np.concatenate((points, constant_term), axis=-1)

    points = np.einsum('nm,...m->...n', transform, points)
    return points[..., :3]


def transform_points_batch(points, transform, translate=True):
    if translate:
        constant_term = torch.ones_like(points[..., :1])
    else:
        constant_term = torch.zeros_like(points[..., :1])
    
    points = torch.cat([points, constant_term], dim=-1)
    points = torch.einsum('bnm,b...m->b...n', transform, points)

    return points[..., :3]


#@ FROM PyTorch3D
class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]:
            ```
            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]
            ```
        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.
        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`
        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`
        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.
        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (x, embed.sin(), embed.cos()
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
        """
        embed = (x[..., None] * self._frequencies * np.pi).view(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool,
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.
        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )