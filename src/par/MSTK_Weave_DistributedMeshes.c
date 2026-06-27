/* 
Copyright 2019 Triad National Security, LLC. All rights reserved.

This file is part of the MSTK project. Please see the license file at
the root of this repository or at
https://github.com/MeshToolkit/MSTK/blob/master/LICENSE
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "MSTK.h"
#include "MSTK_private.h"

#ifdef __cplusplus
extern "C" {
#endif

  static int MSTK_RepairBoundarySideSetPTypes(Mesh_ptr mesh, int rank,
                                              MSTK_Comm comm) {
    int idx = 0, local_repaired = 0, global_repaired = 0;
    MSet_ptr mset;

    while ((mset = MESH_Next_MSet(mesh,&idx))) {
      int eidx = 0;
      MFace_ptr mf;

      if (MSet_EntDim(mset) != MFACE) continue;

      while ((mf = MSet_Next_Entry(mset,&eidx))) {
        List_ptr fregs;
        MRegion_ptr mr;

        if (MF_PType(mf) != PGHOST) continue;

        fregs = MF_Regions(mf);
        if (!fregs || List_Num_Entries(fregs) < 1) {
          if (fregs) List_Delete(fregs);
          continue;
        }

        for (int i = 0; i < List_Num_Entries(fregs); i++) {
          mr = List_Entry(fregs,i);
          if (mr && MR_PType(mr) != PGHOST) {
            MF_Set_PType(mf,MR_PType(mr));
            local_repaired++;
            break;
          }
        }
        List_Delete(fregs);
      }
    }

#ifdef MSTK_HAVE_MPI
    if (comm)
      MPI_Allreduce(&local_repaired,&global_repaired,1,MPI_INT,MPI_SUM,comm);
    else
#endif
      global_repaired = local_repaired;

    if (global_repaired) {
      char mesg[256];
      sprintf(mesg,
              "Repaired %-d boundary side-set face ptypes after weaving",
              global_repaired);
      if (rank == 0)
        MSTK_Report("MSTK_Weave_DistributedMeshes",mesg,MSTK_MESG);
    }

    return global_repaired;
  }

  static int MSTK_DisableBoundarySideSetPTypeRepair(void) {
    const char *val = getenv("MSTK_DISABLE_BOUNDARY_SIDESET_PTYPE_REPAIR");
    return val && val[0] != '\0' && val[0] != '0';
  }



  /* Weave a set of distributed mesh partitions together to build the
     parallel connections and ghost info.

     input_type indicates what info is already present on the mesh
     
     0 -- we are given NO information about how these meshes are connected
          other than the knowledge that they come from the partitioning of
          a single mesh

     1 -- we are given partitioned meshes with a unique global ID on 
          each mesh vertex

     2 -- we are given parallel neighbor information, but no global ID on 
          each mesh vertex


  */
     


  int MSTK_Weave_DistributedMeshes(Mesh_ptr mesh, int topodim,
                                   int num_ghost_layers, int input_type,
                                   MSTK_Comm comm) {

    int have_GIDs = 0;
    int rank, num;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&num);

    if (num_ghost_layers > 1)
      MSTK_Report("MSTK_Weave_DistributedMeshes", "Only 1 ghost layer supported currently", MSTK_FATAL);

    if (input_type > 2) 
      MSTK_Report("MSTK_Weave_DistributedMeshes","Unrecognized input type for meshes", MSTK_WARN);

    // This partition does not have a mesh or has an empty mesh which is ok

    if (mesh == NULL)
      MSTK_Report("MSTK_Weave_DistributedMeshes","MESH is null on this processor",MSTK_FATAL);


    MESH_Set_Prtn(mesh, rank, num);
    
    if (input_type == 0)
      have_GIDs = 0;
    else if (input_type == 1)
      have_GIDs = 1;

    if (input_type == 0 || input_type == 1) {
      /* MESH_MatchEnts_ParBdry(mesh, have_GIDs, rank, num, comm); */
      MESH_AssignGlobalIDs(mesh, topodim, have_GIDs, comm);
      MESH_BuildConnection(mesh, topodim, comm);
    }
    else if (input_type == 2) 
      MESH_AssignGlobalIDs_p2p(mesh, topodim, comm);

    MESH_LabelPType(mesh, topodim, comm);

    if (num_ghost_layers)  /* num_ghost_layers = 0 not well tested */
      MESH_Parallel_AddGhost(mesh, topodim, comm);

    /* Even with no ghost layer of elements, lower dimensional ghosts
     * can exist */
    MESH_Build_GhostLists(mesh, topodim);
                                             

    MESH_Update_ParallelAdj(mesh, comm);
    if (!MSTK_DisableBoundarySideSetPTypeRepair() &&
        MSTK_RepairBoundarySideSetPTypes(mesh, rank, comm))
      MESH_Build_GhostLists(mesh, topodim);
    return 1;
  }


#ifdef __cplusplus
}
#endif
