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

  static int partition_send_timing_enabled(void) {
    const char *val = getenv("MSTK_MESHCONVERT_TIMING");
    return (val && val[0] != '\0' && val[0] != '0');
  }

  static void partition_send_print_time(const char *label, double t0,
                                        double t1, int rank) {
    if (rank == 0)
      fprintf(stderr, "[meshconvert][timing] %-36s %10.3f s\n",
              label, t1 - t0);
  }

  static int partition_send_skip_side_set_attr_copy(void) {
    const char *fast = getenv("MSTK_FAST_SET_COPY");
    if (fast && fast[0] != '\0' && fast[0] != '0')
      return 1;
    fast = getenv("MSTK_ATS_FAST_EXO_SETS");
    if (fast && fast[0] != '\0' && fast[0] != '0')
      return 1;
    const char *val = getenv("MSTK_SKIP_SIDE_SET_ATTR_COPY");
    return (val && val[0] != '\0' && val[0] != '0');
  }

  static int partition_send_batched_set_copy(void) {
    const char *fast = getenv("MSTK_FAST_SET_COPY");
    if (fast && fast[0] != '\0' && fast[0] != '0')
      return 1;
    fast = getenv("MSTK_ATS_FAST_EXO_SETS");
    if (fast && fast[0] != '\0' && fast[0] != '0')
      return 1;
    const char *val = getenv("MSTK_BATCHED_SET_COPY");
    return (val && val[0] != '\0' && val[0] != '0');
  }

  static int partition_send_sparse_sideset_export(void) {
    const char *val = getenv("MSTK_SPARSE_SIDESET_EXPORT");
    return (val && val[0] != '\0' && val[0] != '0');
  }

  static const char *partition_send_attr_type_name(int atttype) {
    switch (atttype) {
    case INT:
      return "INT";
    case DOUBLE:
      return "DOUBLE";
    case VECTOR:
      return "VECTOR";
    case TENSOR:
      return "TENSOR";
    case POINTER:
      return "POINTER";
    default:
      return "UNKNOWN";
    }
  }

  static const char *partition_send_mtype_name(MType mtype) {
    switch (mtype) {
    case MVERTEX:
      return "VERTEX";
    case MEDGE:
      return "EDGE";
    case MFACE:
      return "FACE";
    case MREGION:
      return "REGION";
    case MALLTYPE:
      return "ALL";
    default:
      return "UNKNOWN";
    }
  }

  static int MESH_CopySets_Batched(Mesh_ptr parentmesh, int num,
                                   Mesh_ptr *submeshes, int nset_global,
                                   int *msetids, char (*msetnames)[256],
                                   int timing, int rank) {
    MAttrib_ptr g2latt = MESH_AttribByName(parentmesh,"Global2Local");
    if (!g2latt)
      MSTK_Report("MESH_CopySets_Batched",
                  "Missing Global2Local attribute", MSTK_FATAL);

    long long total_entries = 0;
    long long total_local_entries = 0;
    double t0 = MPI_Wtime();

    for (int m = 0; m < nset_global; m++) {
      MSet_ptr gmset = MESH_MSet(parentmesh,msetids[m]);
      MType mtype = MSet_EntDim(gmset);
      MEntity_ptr gment, lment;
      MSet_ptr *local_sets;
      int idx = 0;

      local_sets = (MSet_ptr *) calloc(num, sizeof(MSet_ptr));
      if (!local_sets)
        MSTK_Report("MESH_CopySets_Batched",
                    "Could not allocate per-set local lookup table",
                    MSTK_FATAL);

      while ((gment = MSet_Next_Entry(gmset,&idx))) {
        List_ptr lmentlist;
        MEnt_Get_AttVal(gment,g2latt,0,0,&lmentlist);
        if (!lmentlist) continue;
        total_entries++;

        int idx2 = 0;
        while ((lment = List_Next_Entry(lmentlist,&idx2))) {
          Mesh_ptr submesh = MEnt_Mesh(lment);

          for (int i = 0; i < num; ++i) {
            if (submesh == submeshes[i]) {
              MSet_ptr lmset = local_sets[i];
              if (!lmset) {
                lmset = MESH_MSetByName(submeshes[i],msetnames[m]);
                if (!lmset)
                  lmset = MSet_New(submeshes[i],msetnames[m],mtype);
                local_sets[i] = lmset;
              }
              MSet_Add(lmset,lment);
              total_local_entries++;
              break;
            }
          }
        }
      }

      free(local_sets);

      if (timing && rank == 0 && (m+1)%10000 == 0)
        fprintf(stderr,
                "[meshconvert][timing] rank0 batched set copy progress sets=%d/%d elapsed=%10.3f s\n",
                m+1, nset_global, MPI_Wtime() - t0);
    }

    if (timing && rank == 0)
      fprintf(stderr,
              "[meshconvert][timing] rank0 batched set copy stats global_entries=%lld local_entries=%lld\n",
              total_entries, total_local_entries);

    return 1;
  }

  /* 
     Partition a mesh into as many submeshes as requested and distribute them

     Author(s): Duo Wang, Rao Garimella
  */


  /* Partition a given mesh into 'num' submeshes, adding a 'ring'
     layers of ghost elements around each partition. If 'with_attr' is
     1, attributes from the mesh are copied onto the submeshes. This
     routine does not send the meshes to other partitions */

  int MESH_Partition_and_Send(Mesh_ptr parentmesh,  int num, int *part, 
                              int *toranks, int ring, int with_attr, 
                              int del_inmesh, MSTK_Comm comm,
                              Mesh_ptr *mysubmesh) {
    int i, j, a, m, idx, ival, rank, numprocs, atttype;
    double rval;
    Mesh_ptr *submeshes = (Mesh_ptr *) malloc(num*sizeof(Mesh_ptr));
    MAttrib_ptr attrib, g2latt, l2gatt;
    MSet_ptr mset;
    MVertex_ptr mv;
    MEdge_ptr me;
    MFace_ptr mf;
    MRegion_ptr mr;
    List_ptr g2llist;
    int numreq=0, maxreq=25, numptrs2free=0, maxptrs2free=25;
    MPI_Request *requests=NULL;
    void **ptrs2free = NULL;
    int maxpendreq = 200;
    int p, n;
    int torank;
    
    char funcname[256] = "MESH_Partition_And_Send";
    int timing = partition_send_timing_enabled();
    int skip_side_set_attrs = partition_send_skip_side_set_attr_copy();
    int batched_set_copy = partition_send_batched_set_copy();
    int sparse_sideset_export = partition_send_sparse_sideset_export();
    double t0, t1;
    int skipped_side_set_attrs = 0;


    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&numprocs);

    /* Check will go away once we let multiple processors do the distribution */
    if (numprocs != num)
      MSTK_Report(funcname,
                  "Number of partitions not equal to number of processors",
                  MSTK_FATAL);

    /* Create an attribute to keep track of the connections from
       entities of the global mesh to entities of submeshes */
    /* This attribute value will be populated with lists in
       MESH_Partition along with another attribute called Local2Global
       and used in MESH_BuildPBoundary and MESH_AddGhost. They are not
       needed subsequently */
    /* NOTE: PROBABLY SHOULD CHANGE IT TO PARENT2CHILD AND CHILD2PARENT */

    g2latt = MAttrib_New(parentmesh,"Global2Local",POINTER,MALLTYPE);


    /* Split the mesh into 'num' submeshes */

    t0 = MPI_Wtime();
    MESH_Partition(parentmesh, num, part, submeshes);
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 MESH_Partition", t0, t1, rank);

    t0 = MPI_Wtime();
    for (i = 0; i < num; i++) {
      /* Tag entities as being in the partition interior or on the
         partition boundary */

      MESH_BuildPBoundary(parentmesh,submeshes[i]);

      /* Add ghost layers */

      MESH_AddGhost(parentmesh,submeshes[i],i,ring);
    }
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 boundary+ghost submeshes", t0, t1,
                                rank);


    /* Send/receive mesh */
    
    requests = (MPI_Request *) malloc(maxreq*sizeof(MPI_Request));
    ptrs2free = (void **) malloc(maxptrs2free*sizeof(void *));
    

    /* Send Mesh Meta Data */
    
    t0 = MPI_Wtime();
    for (n = 0; n < num; n++) {
      torank = toranks[n];
      if (torank == rank) continue;
        
      MESH_Send_MetaData(submeshes[torank], torank, comm,
                         &numreq, &maxreq, &requests,
                         &numptrs2free, &maxptrs2free, &ptrs2free);
        
      /* check if we buffered too many requests - if so, we wait
         until all the data is sent out; if not, we continue. One
         can control how frequently we do a blocking wait for the
         send requests by adjusting maxpendreq */
        
      if (numreq > maxpendreq) {
        if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
          MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
        else {
          numreq = 0;
          for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
          numptrs2free = 0;
        }
      }	
    }
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 send mesh metadata", t0, t1, rank);




    /* Send Mesh Vertices */

    t0 = MPI_Wtime();
    for (n = 0; n < num; n++) {
      torank = toranks[n];
      if (torank == rank) continue;

      MESH_Send_Vertices(submeshes[torank], torank, comm,
                         &numreq, &maxreq, &requests,
                         &numptrs2free, &maxptrs2free, &ptrs2free);
        
      if (numreq > maxpendreq) {
        if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
          MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
        else {
          numreq = 0;
          for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
          numptrs2free = 0;
        }
      }	
    }
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 send vertices", t0, t1, rank);


    /* Send Mesh Vertices */

    t0 = MPI_Wtime();
    for (n = 0; n < num; n++) {
      torank = toranks[n];
      if (torank == rank) continue;

      MESH_Send_VertexCoords(submeshes[torank], torank, comm,
                             &numreq, &maxreq, &requests,
                             &numptrs2free, &maxptrs2free, &ptrs2free);
        
      if (numreq > maxpendreq) {
        if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
          MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
        else {
          numreq = 0;
          for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
          numptrs2free = 0;
        }
      }	
    }
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 send vertex coords", t0, t1, rank);



    /* Send higher dimensional mesh entities  */

    t0 = MPI_Wtime();
    for (n = 0; n < num; n++) {
      torank = toranks[n];
      if (torank == rank) continue;

      MESH_Send_NonVertexEntities(submeshes[torank], torank, comm,
                                  &numreq, &maxreq, &requests,
                                  &numptrs2free, &maxptrs2free, &ptrs2free);
        
      if (numreq > maxpendreq) {
        if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
          MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
        else {
          numreq = 0;
          for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
          numptrs2free = 0;
        }
      }	
    }
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 send non-vertex entities", t0, t1,
                                rank);




    /* If requested, send attributes and mesh sets to partitions */

    if (with_attr) {

      /* First collect attribute information and copy to submeshes */

      int natt_global = MESH_Num_Attribs(parentmesh);
      char (*attnames)[256] = 
        (char (*)[256]) malloc(natt_global*sizeof(char [256]));
      MType side_dim = MESH_Num_Regions(parentmesh) ? MFACE : MEDGE;
      int copied_attrs = 0, copied_side_int_attrs = 0;
      int copied_vertex_attrs = 0, copied_edge_attrs = 0;
      int copied_face_attrs = 0, copied_region_attrs = 0, copied_all_attrs = 0;
      double side_int_attr_time = 0.0;
      double last_attr_progress_time;

      t0 = MPI_Wtime();
      last_attr_progress_time = t0;
      for (a = 0; a < natt_global; a++) {
        double attr_t0, attr_t1;
        MType attdim;
        attrib = MESH_Attrib(parentmesh,a);          

        MAttrib_Get_Name(attrib,attnames[a]);
        if (attrib == g2latt) continue;

        atttype = MAttrib_Get_Type(attrib);
        if (atttype == POINTER) continue;
        attdim = MAttrib_Get_EntDim(attrib);
        if (skip_side_set_attrs && atttype == INT &&
            attdim == side_dim &&
            (sparse_sideset_export ||
             strncmp(attnames[a],"sideset_",8) != 0)) {
          skipped_side_set_attrs++;
          continue;
        }

        attr_t0 = MPI_Wtime();
        MESH_CopyAttr(parentmesh,num,submeshes,attnames[a]);
        attr_t1 = MPI_Wtime();
        copied_attrs++;
        if (atttype == INT && attdim == side_dim) {
          copied_side_int_attrs++;
          side_int_attr_time += attr_t1 - attr_t0;
        }
        switch (attdim) {
        case MVERTEX:
          copied_vertex_attrs++;
          break;
        case MEDGE:
          copied_edge_attrs++;
          break;
        case MFACE:
          copied_face_attrs++;
          break;
        case MREGION:
          copied_region_attrs++;
          break;
        case MALLTYPE:
          copied_all_attrs++;
          break;
        default:
          break;
        }
        if (timing && rank == 0 && attr_t1 - last_attr_progress_time > 10.0) {
          fprintf(stderr,
                  "[meshconvert][timing] rank0 copy attribute progress "
                  "attr=%d/%d copied=%d side-dim-INT=%d elapsed=%10.3f s "
                  "last=%s dim=%s type=%s last-time=%10.3f s\n",
                  a+1, natt_global, copied_attrs, copied_side_int_attrs,
                  attr_t1 - t0, attnames[a],
                  partition_send_mtype_name(attdim),
                  partition_send_attr_type_name(atttype),
                  attr_t1 - attr_t0);
          fflush(stderr);
          last_attr_progress_time = attr_t1;
        }
      }        
      t1 = MPI_Wtime();
      if (timing) {
        partition_send_print_time("rank0 copy attributes", t0, t1, rank);
        if (rank == 0)
          fprintf(stderr,
                  "[meshconvert][timing] rank0 copy attribute counts "
                  "total=%d vertex=%d edge=%d face=%d region=%d all=%d "
                  "side-dim-INT=%d side-dim-INT-time=%10.3f s\n",
                  copied_attrs, copied_vertex_attrs, copied_edge_attrs,
                  copied_face_attrs, copied_region_attrs, copied_all_attrs,
                  copied_side_int_attrs, side_int_attr_time);
        if (skip_side_set_attrs && rank == 0)
          fprintf(stderr,
                  "[meshconvert][timing] skipped side-dim INT attrs=%d\n",
                  skipped_side_set_attrs);
      }

      /* Send Attribute meta data */

      t0 = MPI_Wtime();
      for (n = 0; n < num; n++) {
        torank = toranks[n];
        if (torank == rank) continue;
          
        MESH_Send_AttributeMetaData(submeshes[torank], torank, comm,
                                    &numreq, &maxreq, &requests,
                                    &numptrs2free, &maxptrs2free, &ptrs2free);
          
        if (numreq > maxpendreq) {
          if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
          else {
            numreq = 0;
            for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
            numptrs2free = 0;
          }
        }	
      }
      t1 = MPI_Wtime();
      if (timing)
        partition_send_print_time("rank0 send attribute metadata", t0, t1,
                                  rank);


      /* Send each attribute to the various processors */

      t0 = MPI_Wtime();
      last_attr_progress_time = t0;
      for (a = 0; a < natt_global; a++) {
        double attr_send_t0, attr_send_t1;
        int attr_sends = 0;

        attr_send_t0 = MPI_Wtime();
        for (n = 0; n < num; n++) {
          torank = toranks[n];
          if (torank == rank) continue;

          attrib = MESH_AttribByName(submeshes[torank],attnames[a]);
          if (!attrib) continue; /* this attribute does not exist on this  */
          /* processor - right now its not possible */
          /* but it might be in the future          */
            
          if (MAttrib_Get_Type(attrib) == POINTER) continue;

          MESH_Send_Attribute(submeshes[torank], attrib, torank, comm,
                              &numreq, &maxreq, &requests,
                              &numptrs2free, &maxptrs2free, &ptrs2free);
          attr_sends++;

          if (numreq > maxpendreq) {
            if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
              MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
            else {
              numreq = 0;
              for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
              numptrs2free = 0;
            }
          }	
        }
        attr_send_t1 = MPI_Wtime();
        if (timing && rank == 0 && attr_send_t1 - last_attr_progress_time > 10.0) {
          fprintf(stderr,
                  "[meshconvert][timing] rank0 send attribute progress "
                  "attr=%d/%d sends=%d elapsed=%10.3f s "
                  "last=%s last-time=%10.3f s\n",
                  a+1, natt_global, attr_sends, attr_send_t1 - t0,
                  attnames[a], attr_send_t1 - attr_send_t0);
          fflush(stderr);
          last_attr_progress_time = attr_send_t1;
        }
      }
      t1 = MPI_Wtime();
      if (timing)
        partition_send_print_time("rank0 send attributes", t0, t1, rank);
          
     
        
      /* First collect the mesh set information and copy into submeshes */

      int nset_total = MESH_Num_MSets(parentmesh);
      int nset_global = 0;
      char (*msetnames)[256] = 
        (char (*)[256]) malloc(nset_total*sizeof(char [256]));
      int *msetids = (int *) malloc(nset_total*sizeof(int));

      t0 = MPI_Wtime();
      for (m = 0; m < nset_total; m++) {
        mset = MESH_MSet(parentmesh,m);
        if (sparse_sideset_export && MSet_EntDim(mset) == side_dim)
          continue;
        MSet_Name(mset,msetnames[nset_global]);
        msetids[nset_global] = m;
        nset_global++;
      }
      if (batched_set_copy)
        MESH_CopySets_Batched(parentmesh, num, submeshes, nset_global,
                              msetids, msetnames, timing, rank);
      else
        for (m = 0; m < nset_total; m++) {
          mset = MESH_MSet(parentmesh,m);
          if (sparse_sideset_export && MSet_EntDim(mset) == side_dim)
            continue;
          MESH_CopySet(parentmesh,num,submeshes,mset);
        }
      t1 = MPI_Wtime();
      if (timing)
        fprintf(stderr,
                "[meshconvert][timing] %-36s %10.3f s sets=%d/%d mode=%s\n",
                "rank0 copy mesh sets", t1 - t0, nset_global,
                nset_total,
        batched_set_copy ? "batched" : "per-set");
      free(msetids);
        
      /* Send Mesh Set Meta Data */

      t0 = MPI_Wtime();
      for (n = 0; n < num; n++) {
        torank = toranks[n];
        if (torank == rank) continue;
          
        MESH_Send_MSetMetaData(submeshes[torank], torank, comm,
                               &numreq, &maxreq, &requests,
                               &numptrs2free, &maxptrs2free, &ptrs2free);
          
        /* check if we buffered too many requests - if so, we wait
           until all the data is sent out; if not, we continue. One
           can control how frequently we do a blocking wait for the
           send requests by adjusting maxpendreq */
          
        if (numreq > maxpendreq) {
          if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
          else {
            numreq = 0;
            for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
            numptrs2free = 0;
          }
        }	
      }
      t1 = MPI_Wtime();
      if (timing)
        partition_send_print_time("rank0 send mesh-set metadata", t0, t1,
                                  rank);
        
        
      /* Send Mesh Sets */
        
      t0 = MPI_Wtime();
      if (batched_set_copy) {
        if (numreq) {
          if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
            MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
          else {
            numreq = 0;
            for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
            numptrs2free = 0;
          }
        }

        for (n = 0; n < num; n++) {
          torank = toranks[n];
          if (torank == rank) continue;

          MESH_Send_MSets_Batched(submeshes[torank], torank, comm);
        }
      }
      else {
        for (n = 0; n < num; n++) {
          torank = toranks[n];
          if (torank == rank) continue;

          int nset_local = MESH_Num_MSets(submeshes[torank]);
          for (m = 0; m < nset_local; m++) {
            mset = MESH_MSet(submeshes[torank],m);

            MESH_Send_MSet(submeshes[torank], mset, torank, comm,
                           &numreq, &maxreq, &requests,
                           &numptrs2free, &maxptrs2free, &ptrs2free);

            if (numreq > maxpendreq) {
              if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
                MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
              else {
                numreq = 0;
                for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
                numptrs2free = 0;
              }
            }
          }

        }
      }
      t1 = MPI_Wtime();
      if (timing)
        partition_send_print_time("rank0 send mesh sets", t0, t1, rank);
      free(msetnames);
    }


    t0 = MPI_Wtime();
    if (*mysubmesh == NULL)
      *mysubmesh = submeshes[rank];
    else
      MESH_Copy(submeshes[rank],*mysubmesh,1,1);
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 keep/copy local submesh", t0, t1,
                                rank);


    /* Final flush of all requests */

    t0 = MPI_Wtime();
    if (numreq) {
      if (MPI_Waitall(numreq,requests,MPI_STATUSES_IGNORE) != MPI_SUCCESS)
        MSTK_Report("MSTK_Mesh_Distribute","Could not send mesh",MSTK_FATAL);
      else {
        numreq = 0;
        for (p = 0; p < numptrs2free; ++p) free(ptrs2free[p]);
        numptrs2free = 0;
      }
    }
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 final MPI wait", t0, t1, rank);
      
    if (maxptrs2free) free(ptrs2free);
    if (maxreq) free(requests);


    /* Cleanup */
    /* Delete temporary submeshes */

    if (*mysubmesh != submeshes[rank])
      MESH_Delete(submeshes[rank]);

    for (n = 1; n < num; ++n)
      MESH_Delete(submeshes[n]);
      
    free(submeshes);


    /* Delete the lists associated with g2latt attribute but don't
       remove attribute itself from each of these entities - it will
       get deleted when the mesh gets deleted */

    idx = 0;
    while ((mv = MESH_Next_Vertex(parentmesh,&idx))) {
      MEnt_Get_AttVal(mv,g2latt,&ival,&rval,&g2llist);
      if (g2llist) List_Delete(g2llist);
    }
	
    idx = 0;
    while ((me = MESH_Next_Edge(parentmesh,&idx))) {
      MEnt_Get_AttVal(me,g2latt,&ival,&rval,&g2llist);
      if (g2llist) List_Delete(g2llist);
    }
	
    idx = 0;
    while ((mf = MESH_Next_Face(parentmesh,&idx))) {
      MEnt_Get_AttVal(mf,g2latt,&ival,&rval,&g2llist);
      if (g2llist) List_Delete(g2llist);
    }
	
    idx = 0;
    while ((mr = MESH_Next_Region(parentmesh,&idx))) {
      MEnt_Get_AttVal(mr,g2latt,&ival,&rval,&g2llist);
      if (g2llist) List_Delete(g2llist);
    }	
	
    t0 = MPI_Wtime();
    if (del_inmesh) MESH_Delete(parentmesh);
    t1 = MPI_Wtime();
    if (timing)
      partition_send_print_time("rank0 cleanup/delete parent", t0, t1,
                                rank);
    return 1;
  }

#ifdef __cplusplus
}
#endif
