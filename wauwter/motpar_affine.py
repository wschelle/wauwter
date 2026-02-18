#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:08:02 2024

@author: WauWter
"""

import numpy as np
import sys
from .wauwternifti import readnii
import ants

affinefile=sys.argv[1]
niifile=sys.argv[2]

# affinefile='/home/control/wousch/project/bids7T/sub-014/derivatives/pipe/func/sub-014_task-wmg_run-1_bold_POCS_NORDIC_MCMOCOparams.csv'
# niifile='/home/control/wousch/project/bids7T/sub-014/derivatives/pipe/func/sub-014_task-wmg_run-1_bold_POCS_NORDIC.nii.gz'

readaffine = np.genfromtxt(affinefile, delimiter=',')
readaffine=readaffine[1:,2:]

hdr=readnii(niifile,header_only=True)
# space=np.array(hdr['dim'][1:4])

affmat=np.zeros((readaffine.shape[0],4,4),dtype=np.float64)
for i in range(readaffine.shape[0]):
    affmat[i,:3,:3]=np.reshape(readaffine[i,:9],(3,3))
    affmat[i,:3,3]=readaffine[i,9:]
    affmat[i,3,3]=1

def decompose_affine_matrix(matrix):
    # Ensure it's a 4x4 matrix
    assert matrix.shape == (4, 4), "The input matrix must be 4x4."
    
    # Translation: Extract from the last column
    translation = matrix[:3, 3]

    # Extract the 3x3 rotation+scale+shear matrix
    RS_matrix = matrix[:3, :3]

    # Scale factors (zoom): Magnitude of each column vector of RS_matrix
    scale_factors = np.linalg.norm(RS_matrix, axis=0)

    # Normalize the RS_matrix to remove the scaling component
    rotation_shear_matrix = RS_matrix / scale_factors

    # Compute shear by checking how off-diagonal elements behave
    shear_xy = np.dot(rotation_shear_matrix[:, 0], rotation_shear_matrix[:, 1])
    shear_xz = np.dot(rotation_shear_matrix[:, 0], rotation_shear_matrix[:, 2])
    shear_yz = np.dot(rotation_shear_matrix[:, 1], rotation_shear_matrix[:, 2])

    # Rotation matrix: Use the orthogonal part of the normalized matrix
    rotation_matrix = np.copy(rotation_shear_matrix)
    rotation_matrix[:, 1] -= shear_xy * rotation_matrix[:, 0]
    rotation_matrix[:, 2] -= shear_xz * rotation_matrix[:, 0] + shear_yz * rotation_matrix[:, 1]

    return {
        'translation': translation,
        'scale': scale_factors,
        'shear': (shear_xy, shear_xz, shear_yz),
        'rotation_matrix': rotation_matrix
    }

def rotation_matrix_to_euler_angles(rotation_matrix):
    assert rotation_matrix.shape == (3, 3), "The input matrix must be 3x3."

    # Check for gimbal lock (singularity at cos(theta_y) = 0, where theta_y is rotation around Y-axis)
    if abs(rotation_matrix[2, 0]) != 1:
        # General case
        theta_y = -np.arcsin(rotation_matrix[2, 0])
        cos_theta_y = np.cos(theta_y)
        theta_x = np.arctan2(rotation_matrix[2, 1] / cos_theta_y, rotation_matrix[2, 2] / cos_theta_y)
        theta_z = np.arctan2(rotation_matrix[1, 0] / cos_theta_y, rotation_matrix[0, 0] / cos_theta_y)
    else:
        # Gimbal lock case: cos(theta_y) = 0, so sin(theta_y) = ±1
        theta_z = 0  # Can set Z rotation to zero in this case
        if rotation_matrix[2, 0] == -1:
            theta_y = np.pi / 2
            theta_x = np.arctan2(rotation_matrix[0, 1], rotation_matrix[0, 2])
        else:
            theta_y = -np.pi / 2
            theta_x = np.arctan2(-rotation_matrix[0, 1], -rotation_matrix[0, 2])

    return np.degrees(theta_x), np.degrees(theta_y), np.degrees(theta_z)

# def total_displacement(matrix, point):
#     # Convert point to homogeneous coordinates (add 1 at the end)
#     point_homogeneous = np.append(point, 1)

#     # Apply the affine transformation
#     transformed_point = np.dot(matrix, point_homogeneous)

#     # Compute displacement (difference between original and transformed point)
#     displacement = transformed_point[:3] - point

#     return displacement

def framewise_displacement(T1, T2):
    # Compute the inverse of the first affine matrix
    T1_inv = np.linalg.inv(T1)

    # Compute the relative transformation matrix
    T_rel = np.dot(T1_inv, T2)

    # Extract the translation part from the relative transformation
    relative_translation = T_rel[:3, 3]*hdr['pixdim'][1:4]  # Extract the last column (translation part)

    # Compute the displacement as the Euclidean norm of the translation vector
    displacement = np.linalg.norm(relative_translation)

    return displacement, T_rel

def apply_affine_to_3d_grid(matrix_shape, affine_matrix):
    # Generate the 3D grid coordinates
    z, y, x = np.meshgrid(np.arange(matrix_shape[2])*hdr['pixdim'][3], 
                          np.arange(matrix_shape[1])*hdr['pixdim'][2], 
                          np.arange(matrix_shape[0])*hdr['pixdim'][1], 
                          indexing='ij')

    # Flatten the coordinates into a list of points (x, y, z)
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Convert points to homogeneous coordinates (add a 1 to each point)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the affine transformation to all points
    transformed_points = points_homogeneous @ affine_matrix.T  # Matrix multiplication

    # Calculate the displacement for each point
    displacements = transformed_points[:, :3] - points

    # Reshape the displacements back to the original 3D grid shape
    displacements = displacements.reshape(matrix_shape[2], matrix_shape[1], matrix_shape[0], 3)

    return displacements

translation_rotation=np.zeros((readaffine.shape[0],6),dtype=np.float64)
totrel_displacement=np.zeros((readaffine.shape[0],5),dtype=np.float64)
for i in range(readaffine.shape[0]):
    decaff=decompose_affine_matrix(affmat[i,:,:])
    translation_rotation[i,0:3]=decaff['translation']
    translation_rotation[i,3:]=rotation_matrix_to_euler_angles(decaff['rotation_matrix'])
    totdisps=apply_affine_to_3d_grid(hdr['dim'][1:4],affmat[i,:,:])
    totdisps=np.linalg.norm(totdisps, axis=-1)
    totrel_displacement[i,0]=np.median(totdisps)
    if i>0:
        reldisp,Trel=framewise_displacement(affmat[i-1,:,:],affmat[i,:,:])
        totrel_displacement[i,1]=reldisp
        reldecaff=decompose_affine_matrix(Trel)
        totrel_displacement[i,2:]=rotation_matrix_to_euler_angles(reldecaff['rotation_matrix'])
        
headers1='Tx,Ty,Tz,Rx,Ry,Rz'
headers2='GlobDisp,FWTxyz,FWRx,FWRy,FWRz'

outfilebase=affinefile.split('.csv')[0]
np.savetxt(outfilebase+'_mp.csv', translation_rotation, delimiter=',', newline='\n', header=headers1)
np.savetxt(outfilebase+'_mp_disp.csv', totrel_displacement, delimiter=',', newline='\n', header=headers2)
   
 
# Extract the 3x3 affine part and the translation vector
for i in range(readaffine.shape[0]):
    affine_3x3 = affmat[i,:3,:3].flatten().tolist()
    translation = affmat[i,:3,3].tolist()
    # Create an affine transform using ANTsPyX
    transform = ants.create_ants_transform(transform_type='AffineTransform', dimension=3)
    # Set the parameters for the transform (3x3 matrix + translation)
    parameters = affine_3x3 + translation
    transform.set_parameters(parameters)    
    # Set the fixed parameters (typically the center of rotation, here we assume it's [0, 0, 0])
    transform.set_fixed_parameters([0, 0, 0])
    # Save the transform to a .mat file using the correct ants.write_transform function
    output_file = '/'.join(outfilebase.split('/')[:-1])+'/tmp/'+outfilebase.split('/')[-1]+'_'+f"{i:04d}"+'.mat'
    ants.write_transform(transform, output_file)
