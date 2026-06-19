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
#include "MSTK_private.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Send one mesh set to a particular processor 

     Author: Rao Garimella
  */
  

  int MESH_Send_MSet(Mesh_ptr mesh, MSet_ptr mset, int torank, 
                     MSTK_Comm comm, int *numreq, int *maxreq,
                     MPI_Request **requests, int *numptrs2free, 
                     int *maxptrs2free, void ***ptrs2free) {
    int i, idx;
    int num, nent;
    int *list_info, *list_value_int;
    MType mtype;
    MEntity_ptr ment;
    MPI_Request mpirequest;

    if (requests == NULL)
      MSTK_Report("MSTK_SendMSet","Invalid MPI request buffer",MSTK_FATAL);
  
    if (*maxreq == 0) {
      *maxreq = 10;
      *requests = (MPI_Request *) malloc(*maxreq*sizeof(MPI_Request));
      *numreq = 0;
    }
    else if (*maxreq < (*numreq) + 2) {
      *maxreq *= 2;
      *requests = (MPI_Request *) realloc(*requests,*maxreq*sizeof(MPI_Request));
    }
  
    mtype = MSet_EntDim(mset);
    nent = MSet_Num_Entries(mset);

    list_info = (int *)malloc(sizeof(int));
    list_info[0] = nent;

    MPI_Isend(list_info,1,MPI_INT,torank,torank,comm,&mpirequest);
    (*requests)[*numreq] = mpirequest;
    (*numreq)++;

    if (!nent) {
      free(list_info);
      return 1;
    }

    /* attribute index and global id */ 
    num = 2*nent;
    list_value_int = (int *)malloc(num*sizeof(int));

    idx = 0; i = 0;
    while ((ment = MSet_Next_Entry(mset,&idx))) {
      list_value_int[i++] = MEnt_Dim(ment);
      list_value_int[i++] = MEnt_GlobalID(ment);
    }
  

    /* send info */
    MPI_Isend(list_value_int,num,MPI_INT,torank,torank,comm,&mpirequest);
    (*requests)[*numreq] = mpirequest;
    (*numreq)++;

    /* track the buffers used for sending */

    if (*maxptrs2free == 0) {
      *maxptrs2free = 25;
      *ptrs2free = (void **) malloc(*maxptrs2free*sizeof(void *));
      *numptrs2free = 0;
    }
    else if (*maxptrs2free < (*numptrs2free) + 2) {
      *maxptrs2free = 2*(*maxptrs2free) + 2 ;
      *ptrs2free = (void **) realloc(*ptrs2free,(*maxptrs2free)*sizeof(void *));
    }

    (*ptrs2free)[(*numptrs2free)++] = list_info;
    (*ptrs2free)[(*numptrs2free)++] = list_value_int;  
  
    return 1;
  }

  int MESH_Send_MSets_Batched(Mesh_ptr mesh, int torank, MSTK_Comm comm) {
    int m, idx, nset, total_entries, pair_count, result;
    int *header, *offsets, *entry_pairs;
    MSet_ptr mset;
    MEntity_ptr ment;

    nset = MESH_Num_MSets(mesh);
    total_entries = 0;
    for (m = 0; m < nset; m++) {
      mset = MESH_MSet(mesh,m);
      total_entries += MSet_Num_Entries(mset);
    }

    header = (int *) malloc(2*sizeof(int));
    offsets = (int *) malloc((nset+1)*sizeof(int));
    entry_pairs = total_entries ?
      (int *) malloc(2*total_entries*sizeof(int)) : NULL;

    if (!header || !offsets || (total_entries && !entry_pairs))
      MSTK_Report("MESH_Send_MSets_Batched",
                  "Could not allocate batched mesh set buffers",
                  MSTK_FATAL);

    header[0] = nset;
    header[1] = total_entries;

    pair_count = 0;
    offsets[0] = 0;
    for (m = 0; m < nset; m++) {
      mset = MESH_MSet(mesh,m);
      idx = 0;
      while ((ment = MSet_Next_Entry(mset,&idx))) {
        entry_pairs[2*pair_count] = MEnt_Dim(ment);
        entry_pairs[2*pair_count+1] = MEnt_GlobalID(ment);
        pair_count++;
      }
      offsets[m+1] = pair_count;
    }

    result = MPI_Send(header,2,MPI_INT,torank,torank,comm);
    if (result != MPI_SUCCESS)
      MSTK_Report("MESH_Send_MSets_Batched",
                  "Error sending mesh set batch header", MSTK_FATAL);

    result = MPI_Send(offsets,nset+1,MPI_INT,torank,torank,comm);
    if (result != MPI_SUCCESS)
      MSTK_Report("MESH_Send_MSets_Batched",
                  "Error sending mesh set batch offsets", MSTK_FATAL);

    if (total_entries) {
      result = MPI_Send(entry_pairs,2*total_entries,MPI_INT,torank,torank,
                        comm);
      if (result != MPI_SUCCESS)
        MSTK_Report("MESH_Send_MSets_Batched",
                    "Error sending mesh set batch entries", MSTK_FATAL);
    }

    free(header);
    free(offsets);
    free(entry_pairs);

    return 1;
  }


  
#ifdef __cplusplus
}
#endif


