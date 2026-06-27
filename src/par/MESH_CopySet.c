/* 
Copyright 2019 Triad National Security, LLC. All rights reserved.

This file is part of the MSTK project. Please see the license file at
the root of this repository or at
https://github.com/MeshToolkit/MSTK/blob/master/LICENSE
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "MSTK.h"

#ifdef __cplusplus
extern "C" {
#endif

  static int sparse_set_copy_enabled(void) {
    const char *fast = getenv("MSTK_FAST_SET_COPY");
    if (fast && fast[0] != '\0' && fast[0] != '0')
      return 1;
    fast = getenv("MSTK_ATS_FAST_EXO_SETS");
    if (fast && fast[0] != '\0' && fast[0] != '0')
      return 1;
    const char *val = getenv("MSTK_SPARSE_SET_COPY");
    return (val && val[0] != '\0' && val[0] != '0');
  }

 /* 
    this function copy set information from mesh to submesh
    mset_global - Set in the global_mesh
    Create the set in the submesh if it is not there

 */

  int MESH_CopySet(Mesh_ptr mesh, int num, Mesh_ptr *submeshes, MSet_ptr gmset) {
  int i, idx, idx2, found;
  MType mtype;
  MEntity_ptr lment, gment;
  MSet_ptr lmset;
  char msetname[256];
  void *loc;
  MAttrib_ptr g2latt;
  List_ptr lmentlist;
  MSet_ptr *lmset_array;
  Mesh_ptr submesh;
  int sparse_copy = sparse_set_copy_enabled();

  lmset_array = (MSet_ptr *) calloc(num,sizeof(MSet_ptr));

  g2latt = MESH_AttribByName(mesh,"Global2Local");

  MSet_Name(gmset,msetname);
  mtype = MSet_EntDim(gmset);

  if (!sparse_copy) {
    for (i = 0; i < num; ++i) {
      lmset_array[i] = MESH_MSetByName(submeshes[i],msetname);
      if (!lmset_array[i])
        lmset_array[i] = MSet_New(submeshes[i],msetname,mtype);
    }
  }
 
  

  idx = 0;
  while ((gment = MSet_Next_Entry(gmset,&idx))) {
    MEnt_Get_AttVal(gment,g2latt,0,0,&lmentlist);
    
    idx2 = 0;
    while ((lment = List_Next_Entry(lmentlist,&idx2))) {
      submesh = MEnt_Mesh(lment);

      for (i = 0; i < num; ++i) {
	if (submesh == submeshes[i]) {
	  lmset = lmset_array[i];
          if (!lmset) {
            lmset = MESH_MSetByName(submeshes[i],msetname);
            if (!lmset)
              lmset = MSet_New(submeshes[i],msetname,mtype);
            lmset_array[i] = lmset;
          }
	  MSet_Add(lmset,lment);
	  break;
	}
      }

    }
  }

  free(lmset_array);
  return 1;
}
  
#ifdef __cplusplus
}
#endif
