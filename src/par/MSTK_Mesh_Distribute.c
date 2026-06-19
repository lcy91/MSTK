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

  static int distribute_timing_enabled(void) {
    const char *val = getenv("MSTK_MESHCONVERT_TIMING");
    return (val && val[0] != '\0' && val[0] != '0');
  }

  static void distribute_print_time(const char *label, double dt, int rank,
                                    MSTK_Comm comm) {
    double dt_min = 0.0, dt_sum = 0.0, dt_max = 0.0;
    int nproc = 1;
    MPI_Comm_size(comm, &nproc);
    MPI_Reduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&dt, &dt_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&dt, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][timing] %-36s min %10.3f s mean %10.3f s max %10.3f s\n",
              label, dt_min, dt_sum/nproc, dt_max);
  }


  /* Partition a given mesh and distribute it to 'num' processors 

     Authors: Rao Garimella
              Duo Wang
  */


  int MSTK_Mesh_Distribute(Mesh_ptr parentmesh, Mesh_ptr *mysubmesh, int *dim, 
			   int ring, int with_attr, int method, 
			   int del_inmesh, MSTK_Comm comm) {
    int i, a, m, n, recv_dim;
    int *send_dim, *part=NULL;
    int rank, numprocs, *toranks;
    int DebugWait=0;
    MAttrib_ptr attrib;
    MSet_ptr mset;
    int timing = distribute_timing_enabled();
    double partition_dt = 0.0, send_dt = 0.0, recv_dt = 0.0;
    double ghost_dt = 0.0, adj_dt = 0.0;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&numprocs);
    recv_dim = rank+5;          /* ??? */

    while (DebugWait) ;

#ifdef DEBUG
    double elapsed_time;
    double t0 = MPI_Wtime();
#endif

    send_dim = (int *) malloc(numprocs*sizeof(int));
    for (i = 0; i < numprocs; i++) send_dim[i] = *dim;

    MPI_Scatter(send_dim, 1, MPI_INT, &recv_dim, 1, MPI_INT, 0, comm);
    free(send_dim);
    if (rank != 0)
      *dim = recv_dim;


    /* PARTITIONING OF THE MESH GRAPH AND GETTING THE PARTITION
       NUMBERS FOR EACH ELEMENT */

    double partition_t0 = MPI_Wtime();
    MESH_Get_Partitioning(parentmesh, method, &part, comm);
    partition_dt = MPI_Wtime() - partition_t0;
    if (timing)
      distribute_print_time("MESH_Get_Partitioning", partition_dt, rank, comm);


    if (rank == 0) {
      toranks = (int *) malloc(numprocs*sizeof(int));
      for (i = 0; i < numprocs; i++) toranks[i] = i;

      if (*mysubmesh == NULL)
        *mysubmesh = MESH_New(MESH_RepType(parentmesh));
      double send_t0 = MPI_Wtime();
      MESH_Partition_and_Send(parentmesh, numprocs, part, toranks, ring, 
                              with_attr, del_inmesh, comm, mysubmesh);
      send_dt = MPI_Wtime() - send_t0;

      free(toranks);

      fprintf(stderr,"Finished partioning and sending on rank 0\n");
    }

    if (part) free(part);

    
    /* RECEIVING THE INFORMATION ON OTHER PROCESSORS - THIS IS
       STRAIGHTFORWARD AND USES BLOCKING MPI RECEIVES BECAUSE EACH
       STEP NEEDS TO BE COMPLETED BEFORE THE NEXT */


    if (rank > 0) { /* Receive the mesh from processor 0 */

      int fromrank = 0;
      int nv, ne, nf, nr;
      RepType rtype;

      if (*mysubmesh == NULL)
        *mysubmesh = MESH_New(UNKNOWN_REP);
      double recv_t0 = MPI_Wtime();
      MESH_RecvMesh(*mysubmesh, fromrank, with_attr, comm);
      recv_dt = MPI_Wtime() - recv_t0;

    }
    if (timing) {
      distribute_print_time("rank0 partition/send", send_dt, rank, comm);
      distribute_print_time("worker receive mesh", recv_dt, rank, comm);
    }



    /* FINISH UP */
    /* Build the sorted lists of ghost and overlap entities */

    double ghost_t0 = MPI_Wtime();
    MESH_Build_GhostLists(*mysubmesh,*dim);
    ghost_dt = MPI_Wtime() - ghost_t0;


    /* Some additional parallel information update to indicate which
       processors communicate with which others */
    
    MESH_Set_Prtn(*mysubmesh,rank,numprocs);

    double adj_t0 = MPI_Wtime();
    MESH_Update_ParallelAdj(*mysubmesh,comm);
    adj_dt = MPI_Wtime() - adj_t0;
    if (timing) {
      distribute_print_time("MESH_Build_GhostLists", ghost_dt, rank, comm);
      distribute_print_time("MESH_Update_ParallelAdj", adj_dt, rank, comm);
    }


    MESH_Disable_GlobalIDSearch(*mysubmesh);


#ifdef DEBUG
    elapsed_time = MPI_Wtime() - t0;
    fprintf(stderr,"Elapsed time after mesh distribution on processor %-d is %lf s\n",rank,elapsed_time);
#endif


    /* Put a barrier so that distribution of meshes takes place one at a time 
       in a simulation that may have multiple mesh objects on each processor */

    MPI_Barrier(comm);

    return 1;
  }



#ifdef __cplusplus
}
#endif
