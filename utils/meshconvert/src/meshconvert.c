/* 
Copyright 2019 Triad National Security, LLC. All rights reserved.

This file is part of the MSTK project. Please see the license file at
the root of this repository or at
https://github.com/MeshToolkit/MSTK/blob/master/LICENSE
*/

/* Translator between mesh formats */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/resource.h>
#include <sys/time.h>

#ifdef _MSTK_HAVE_MPI
#include <mpi.h>
#endif

#include "MSTK.h"
#include "MSTK_private.h"
#include "exodusII.h"

#define MSTK_ATS_ORIG_ELEM_GID_ATT "MSTK_ATS_ORIG_ELEM_GID"

extern int FixColumnPartitions_UpDownFaces(Mesh_ptr mesh, MRegion_ptr mr,
                                           MFace_ptr *up, MFace_ptr *down);

static double meshconvert_wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return ((double) tv.tv_sec) + 1.0e-6*((double) tv.tv_usec);
}

static double meshconvert_maxrss_mb(void) {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    return 0.0;
#ifdef __APPLE__
  return ((double) usage.ru_maxrss)/(1024.0*1024.0);
#else
  return ((double) usage.ru_maxrss)/1024.0;
#endif
}

static void meshconvert_print_time(const char *label, double t0, double t1,
                                   int rank, MSTK_Comm comm) {
  double dt = t1 - t0;
#ifdef MSTK_HAVE_MPI
  if (comm) {
    double dt_min = 0.0, dt_sum = 0.0, dt_max = 0.0;
    MPI_Reduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&dt, &dt_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&dt, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int nproc = 1;
    MPI_Comm_size(comm, &nproc);
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][timing] %-36s min %10.3f s mean %10.3f s max %10.3f s\n",
              label, dt_min, dt_sum/nproc, dt_max);
    return;
  }
#endif
  if (rank == 0)
    fprintf(stderr, "[meshconvert][timing] %-36s %10.3f s\n", label, dt);
}

static void meshconvert_print_memory(const char *label, int rank, MSTK_Comm comm) {
  double rss = meshconvert_maxrss_mb();
#ifdef MSTK_HAVE_MPI
  if (comm) {
    double rss_min = 0.0, rss_sum = 0.0, rss_max = 0.0;
    MPI_Reduce(&rss, &rss_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&rss, &rss_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&rss, &rss_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    int nproc = 1;
    MPI_Comm_size(comm, &nproc);
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][memory] %-36s min %10.1f MB mean %10.1f MB max %10.1f MB\n",
              label, rss_min, rss_sum/nproc, rss_max);
    return;
  }
#endif
  if (rank == 0)
    fprintf(stderr, "[meshconvert][memory] %-36s %10.1f MB\n", label, rss);
}

static int meshconvert_enable_unsafe_ats_exo_workflow(void) {
  const char *val = getenv("MSTK_ENABLE_UNSAFE_ATS_EXO_WORKFLOW");
  return (val && val[0] != '\0' && val[0] != '0');
}

static const char *meshconvert_ptype_name(PType ptype) {
  switch (ptype) {
  case PINTERIOR: return "PINTERIOR";
  case POVERLAP: return "POVERLAP";
  case PBOUNDARY: return "PBOUNDARY";
  case PGHOST: return "PGHOST";
  default: return "UNKNOWN";
  }
}

static const char *meshconvert_mtype_name(MType mtype) {
  switch (mtype) {
  case MVERTEX: return "MVERTEX";
  case MEDGE: return "MEDGE";
  case MFACE: return "MFACE";
  case MREGION: return "MREGION";
  case MALLTYPE: return "MALLTYPE";
  default: return "UNKNOWN";
  }
}

static int meshconvert_ats_parallel_kind(MEntity_ptr ent) {
#ifdef MSTK_HAVE_MPI
  return MEnt_PType(ent) == PGHOST ? 1 : 0;
#else
  return 0;
#endif
}

static const char *meshconvert_ats_parallel_kind_name(int ptype) {
  return ptype ? "GHOST" : "OWNED";
}

static int meshconvert_check_column_set(Mesh_ptr mesh, const char *setname,
                                        int repair, int rank, MSTK_Comm comm) {
  MSet_ptr mset;
  MFace_ptr mf;
  int idx = 0, local_faces = 0, local_bad = 0, local_repaired = 0;
  int global_faces = 0, global_bad = 0, global_repaired = 0;

  if (!setname || setname[0] == '\0')
    return 1;

  mset = MESH_MSetByName(mesh,setname);
  if (!mset) {
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][verify-column-set] set '%s' not found\n",
              setname);
    return 0;
  }
  if (MSet_EntDim(mset) != MFACE) {
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][verify-column-set] set '%s' is not a face set\n",
              setname);
    return 0;
  }

  while ((mf = MSet_Next_Entry(mset,&idx))) {
    List_ptr fregs = MF_Regions(mf);
    MRegion_ptr mr = NULL;
    int fptype, rptype;

    local_faces++;
    if (!fregs || List_Num_Entries(fregs) < 1) {
      local_bad++;
      if (local_bad == 1)
        fprintf(stderr,
                "[meshconvert][verify-column-set] rank %d face LID=%d GID=%d has no adjacent cell\n",
                rank, MF_ID(mf), MF_GlobalID(mf));
      if (fregs) List_Delete(fregs);
      continue;
    }

    mr = List_Entry(fregs,0);
    fptype = meshconvert_ats_parallel_kind((MEntity_ptr) mf);
    rptype = meshconvert_ats_parallel_kind((MEntity_ptr) mr);
    if (fptype != rptype) {
      if (repair) {
        MF_Set_PType(mf, MR_PType(mr));
        local_repaired++;
        fptype = rptype;
      }
    }
    if (fptype != rptype) {
      local_bad++;
      if (local_bad == 1)
        fprintf(stderr,
                "[meshconvert][verify-column-set] rank %d set '%s' top face LID=%d GID=%d ptype=%s raw=%s first-cell LID=%d GID=%d ptype=%s raw=%s\n",
                rank, setname, MF_ID(mf), MF_GlobalID(mf),
                meshconvert_ats_parallel_kind_name(fptype),
                meshconvert_ptype_name(MF_PType(mf)),
                MR_ID(mr), MR_GlobalID(mr),
                meshconvert_ats_parallel_kind_name(rptype),
                meshconvert_ptype_name(MR_PType(mr)));
      List_Delete(fregs);
      continue;
    }
    List_Delete(fregs);

    while (mr) {
      MFace_ptr upf = NULL, downf = NULL;
      MRegion_ptr next_mr = NULL;
      List_ptr down_regs = NULL;
      int down_ptype;

      FixColumnPartitions_UpDownFaces(mesh,mr,&upf,&downf);
      if (!downf) break;

      down_ptype = meshconvert_ats_parallel_kind((MEntity_ptr) downf);
      if (down_ptype != fptype) {
        if (repair) {
          if (fptype)
            MF_Set_PType(downf, PGHOST);
          else if (MF_PType(downf) == PGHOST)
            MF_Set_PType(downf, POVERLAP);
          local_repaired++;
          down_ptype = fptype;
        }
      }
      if (down_ptype != fptype) {
        local_bad++;
        if (local_bad == 1)
          fprintf(stderr,
                  "[meshconvert][verify-column-set] rank %d set '%s' down face LID=%d GID=%d ptype=%s raw=%s column-top face GID=%d ptype=%s\n",
                  rank, setname, MF_ID(downf), MF_GlobalID(downf),
                  meshconvert_ats_parallel_kind_name(down_ptype),
                  meshconvert_ptype_name(MF_PType(downf)),
                  MF_GlobalID(mf),
                  meshconvert_ats_parallel_kind_name(fptype));
        break;
      }

      down_regs = MF_Regions(downf);
      if (down_regs && List_Num_Entries(down_regs) == 2) {
        MRegion_ptr r0 = List_Entry(down_regs,0);
        MRegion_ptr r1 = List_Entry(down_regs,1);
        next_mr = (r0 == mr) ? r1 : r0;
        if (next_mr &&
            meshconvert_ats_parallel_kind((MEntity_ptr) next_mr) != fptype) {
          local_bad++;
          if (local_bad == 1)
            fprintf(stderr,
                    "[meshconvert][verify-column-set] rank %d set '%s' column cell LID=%d GID=%d ptype=%s raw=%s column-top face GID=%d ptype=%s\n",
                    rank, setname, MR_ID(next_mr), MR_GlobalID(next_mr),
                    meshconvert_ats_parallel_kind_name(
                      meshconvert_ats_parallel_kind((MEntity_ptr) next_mr)),
                    meshconvert_ptype_name(MR_PType(next_mr)),
                    MF_GlobalID(mf),
                    meshconvert_ats_parallel_kind_name(fptype));
          if (down_regs) List_Delete(down_regs);
          break;
        }
      }

      if (down_regs) List_Delete(down_regs);
      mr = next_mr;
    }
  }

#ifdef MSTK_HAVE_MPI
  if (comm) {
    MPI_Reduce(&local_faces,&global_faces,1,MPI_INT,MPI_SUM,0,comm);
    MPI_Reduce(&local_bad,&global_bad,1,MPI_INT,MPI_SUM,0,comm);
    MPI_Reduce(&local_repaired,&global_repaired,1,MPI_INT,MPI_SUM,0,comm);
  } else
#endif
  {
    global_faces = local_faces;
    global_bad = local_bad;
    global_repaired = local_repaired;
  }

  if (rank == 0)
    fprintf(stderr,
            "[meshconvert][verify-column-set] set '%s' faces=%d mismatched-face-cell-ptype=%d repaired=%d\n",
            setname, global_faces, global_bad, global_repaired);

  return global_bad == 0;
}

static unsigned long long meshconvert_hash_gid(unsigned long long hash,
                                               int dim, int gid, int ptype) {
  unsigned long long x = (unsigned long long) (unsigned int) gid;
  x ^= ((unsigned long long) (unsigned int) dim) << 32;
  x ^= ((unsigned long long) (unsigned int) ptype) << 48;
  hash ^= x + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
  return hash;
}

static int meshconvert_summarize_set(Mesh_ptr mesh, const char *setname,
                                     int rank, int numprocs, MSTK_Comm comm) {
  MSet_ptr mset;
  MEntity_ptr ment;
  int idx = 0;
  int local_count = 0, local_nonghost = 0;
  unsigned long long local_hash = 1469598103934665603ULL;
  unsigned long long local_nonghost_hash = 1469598103934665603ULL;

  if (!setname || setname[0] == '\0')
    return 1;

  mset = MESH_MSetByName(mesh,setname);
  if (!mset) {
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][summarize-set] set '%s' not found\n",
              setname);
    return 0;
  }

  while ((ment = MSet_Next_Entry(mset,&idx))) {
    int dim = MEnt_Dim(ment);
    int gid = MEnt_GlobalID(ment);
    int ptype = MEnt_PType(ment);
    local_count++;
    local_hash = meshconvert_hash_gid(local_hash,dim,gid,ptype);
    if (ptype != PGHOST) {
      local_nonghost++;
      local_nonghost_hash =
        meshconvert_hash_gid(local_nonghost_hash,dim,gid,ptype);
    }
  }

#ifdef MSTK_HAVE_MPI
  if (comm) {
    int *counts = NULL, *nonghost_counts = NULL;
    unsigned long long *hashes = NULL, *nonghost_hashes = NULL;
    int global_count = 0, global_nonghost = 0;

    if (rank == 0) {
      counts = (int *) calloc(numprocs,sizeof(int));
      nonghost_counts = (int *) calloc(numprocs,sizeof(int));
      hashes = (unsigned long long *)
        calloc(numprocs,sizeof(unsigned long long));
      nonghost_hashes = (unsigned long long *)
        calloc(numprocs,sizeof(unsigned long long));
    }
    MPI_Gather(&local_count,1,MPI_INT,counts,1,MPI_INT,0,comm);
    MPI_Gather(&local_nonghost,1,MPI_INT,nonghost_counts,1,MPI_INT,0,comm);
    MPI_Gather(&local_hash,1,MPI_UNSIGNED_LONG_LONG,
               hashes,1,MPI_UNSIGNED_LONG_LONG,0,comm);
    MPI_Gather(&local_nonghost_hash,1,MPI_UNSIGNED_LONG_LONG,
               nonghost_hashes,1,MPI_UNSIGNED_LONG_LONG,0,comm);

    if (rank == 0) {
      int i;
      fprintf(stderr,
              "[meshconvert][summarize-set] set '%s' dim=%s ranks=%d\n",
              setname, meshconvert_mtype_name(MSet_EntDim(mset)), numprocs);
      for (i = 0; i < numprocs; i++) {
        global_count += counts[i];
        global_nonghost += nonghost_counts[i];
        fprintf(stderr,
                "[meshconvert][summarize-set] rank=%d count=%d nonghost=%d hash=%016llx nonghost-hash=%016llx\n",
                i, counts[i], nonghost_counts[i], hashes[i],
                nonghost_hashes[i]);
      }
      fprintf(stderr,
              "[meshconvert][summarize-set] global count=%d nonghost=%d\n",
              global_count, global_nonghost);
      free(counts);
      free(nonghost_counts);
      free(hashes);
      free(nonghost_hashes);
    }
  } else
#endif
  {
    fprintf(stderr,
            "[meshconvert][summarize-set] rank=%d set='%s' count=%d nonghost=%d hash=%016llx nonghost-hash=%016llx\n",
            rank, setname, local_count, local_nonghost, local_hash,
            local_nonghost_hash);
  }

  return 1;
}

static int meshconvert_verify_column_set(Mesh_ptr mesh, const char *setname,
                                         int rank, MSTK_Comm comm) {
  return meshconvert_check_column_set(mesh,setname,0,rank,comm);
}

static int meshconvert_repair_column_set(Mesh_ptr mesh, const char *setname,
                                         int rank, MSTK_Comm comm) {
  return meshconvert_check_column_set(mesh,setname,1,rank,comm);
}

static int meshconvert_repair_sparse_column_set(
    Mesh_ptr mesh, const char *exo_file, const char *setname,
    int rank, MSTK_Comm comm) {
  MAttrib_ptr orig_gid_att;
  MRegion_ptr mr;
  int idx = 0, local_max_gid = 0, global_max_gid = 0;
  int local_repaired = 0, global_repaired = 0;
  int local_seen = 0, global_seen = 0;
  MRegion_ptr *orig_gid_to_region = NULL;
  int exoid, cpu_ws, io_ws, status, num_ids, target_id = -1;
  int *ids = NULL, num_sides = 0, num_df = 0;
  int *elem_list = NULL, *side_list = NULL;
  float version;

  if (!mesh || !exo_file || !setname || setname[0] == '\0')
    return 1;

  orig_gid_att = MESH_AttribByName(mesh,MSTK_ATS_ORIG_ELEM_GID_ATT);
  if (!orig_gid_att) {
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][repair-sparse-column-set] missing %s attribute\n",
              MSTK_ATS_ORIG_ELEM_GID_ATT);
    return 0;
  }

  idx = 0;
  while ((mr = MESH_Next_Region(mesh,&idx))) {
    int gid = 0;
    double rval;
    void *pval;
    if (MR_PType(mr) == PGHOST) continue;
    if (MEnt_Get_AttVal(mr,orig_gid_att,&gid,&rval,&pval) && gid > local_max_gid)
      local_max_gid = gid;
  }

#ifdef MSTK_HAVE_MPI
  if (comm)
    MPI_Allreduce(&local_max_gid,&global_max_gid,1,MPI_INT,MPI_MAX,comm);
  else
#endif
    global_max_gid = local_max_gid;

  orig_gid_to_region = global_max_gid ?
    (MRegion_ptr *) calloc(global_max_gid+1,sizeof(MRegion_ptr)) : NULL;
  if (global_max_gid && !orig_gid_to_region)
    MSTK_Report("meshconvert",
                "Could not allocate sparse column repair region map",
                MSTK_FATAL);

  idx = 0;
  while ((mr = MESH_Next_Region(mesh,&idx))) {
    int gid = 0;
    double rval;
    void *pval;
    if (MR_PType(mr) == PGHOST) continue;
    if (MEnt_Get_AttVal(mr,orig_gid_att,&gid,&rval,&pval) &&
        gid > 0 && gid <= global_max_gid)
      orig_gid_to_region[gid] = mr;
  }

  cpu_ws = sizeof(double);
  io_ws = sizeof(double);
  exoid = ex_open(exo_file, EX_READ, &cpu_ws, &io_ws, &version);
  if (exoid < 0) {
    if (orig_gid_to_region) free(orig_gid_to_region);
    MSTK_Report("meshconvert",
                "Could not open Exodus file for sparse column repair",
                MSTK_FATAL);
  }

  num_ids = ex_inquire_int(exoid, EX_INQ_SIDE_SETS);
  ids = num_ids ? (int *) malloc(num_ids*sizeof(int)) : NULL;
  if (num_ids && !ids)
    MSTK_Report("meshconvert",
                "Could not allocate sparse column repair side-set ids",
                MSTK_FATAL);
  if (num_ids) {
    int i;
    status = ex_get_ids(exoid, EX_SIDE_SET, ids);
    if (status < 0)
      MSTK_Report("meshconvert",
                  "Could not read side-set ids for sparse column repair",
                  MSTK_FATAL);
    for (i = 0; i < num_ids; i++) {
      char name[256];
      name[0] = '\0';
      status = ex_get_name(exoid, EX_SIDE_SET, ids[i], name);
      if (status != 0 || name[0] == '\0')
        sprintf(name,"sideset_%-d",ids[i]);
      if (strcmp(name,setname) == 0) {
        target_id = ids[i];
        break;
      }
    }
  }

  if (target_id < 0) {
    if (rank == 0)
      fprintf(stderr,
              "[meshconvert][repair-sparse-column-set] set '%s' not found in %s\n",
              setname, exo_file);
    ex_close(exoid);
    free(ids);
    free(orig_gid_to_region);
    return 0;
  }

  status = ex_get_set_param(exoid, EX_SIDE_SET, target_id,
                            &num_sides, &num_df);
  if (status < 0)
    MSTK_Report("meshconvert",
                "Could not read sparse column repair side-set size",
                MSTK_FATAL);

  elem_list = num_sides ? (int *) malloc(num_sides*sizeof(int)) : NULL;
  side_list = num_sides ? (int *) malloc(num_sides*sizeof(int)) : NULL;
  if (num_sides && (!elem_list || !side_list))
    MSTK_Report("meshconvert",
                "Could not allocate sparse column repair entries",
                MSTK_FATAL);

  if (num_sides) {
    int i;
    status = ex_get_set(exoid, EX_SIDE_SET, target_id, elem_list, side_list);
    if (status < 0)
      MSTK_Report("meshconvert",
                  "Could not read sparse column repair side-set entries",
                  MSTK_FATAL);

    for (i = 0; i < num_sides; i++) {
      int gid = elem_list[i];
      int side = side_list[i];
      MFace_ptr mf = NULL;
      List_ptr rfaces = NULL;
      if (gid <= 0 || gid > global_max_gid) continue;
      mr = orig_gid_to_region[gid];
      if (!mr) continue;
      rfaces = MR_Faces(mr);
      if (rfaces && side > 0 && side <= List_Num_Entries(rfaces))
        mf = List_Entry(rfaces,side-1);
      if (rfaces) List_Delete(rfaces);
      if (!mf) continue;

      local_seen++;
      if (meshconvert_ats_parallel_kind((MEntity_ptr) mf) !=
          meshconvert_ats_parallel_kind((MEntity_ptr) mr)) {
        MF_Set_PType(mf,MR_PType(mr));
        local_repaired++;
      }
    }
  }

#ifdef MSTK_HAVE_MPI
  if (comm) {
    MPI_Reduce(&local_seen,&global_seen,1,MPI_INT,MPI_SUM,0,comm);
    MPI_Reduce(&local_repaired,&global_repaired,1,MPI_INT,MPI_SUM,0,comm);
  } else
#endif
  {
    global_seen = local_seen;
    global_repaired = local_repaired;
  }

  if (rank == 0)
    fprintf(stderr,
            "[meshconvert][repair-sparse-column-set] set '%s' entries-seen=%d repaired=%d\n",
            setname, global_seen, global_repaired);

  ex_close(exoid);
  free(ids);
  free(elem_list);
  free(side_list);
  free(orig_gid_to_region);

  return 1;
}

MshFmt getFormat(char *filename) {
  int len = strlen(filename);
  if (len > 5 && strncmp(&(filename[len-5]),".mstk",5) == 0)
    return MSTK;
  else if (len > 4 && strncmp(&(filename[len-4]),".gmv",4) == 0)
    return GMV;
  else if ((len > 4 && strncmp(&(filename[len-4]),".exo",4) == 0) ||
	   (len > 2 && strncmp(&(filename[len-2]),".g",2) == 0) ||
	   (len > 2 && strncmp(&(filename[len-2]),".e",2) == 0))
    return EXODUSII;
  else if (len > 4 && strncmp(&(filename[len-4]),".par",4) == 0) 
    return NEMESISI;
  else if ((len > 4 && strncmp(&(filename[len-4]),".avs",4) == 0) ||
	   (len > 4 && strncmp(&(filename[len-4]),".inp",4) == 0))
    return AVSUCD;
  else if (len > 4 && strncmp(&(filename[len-4]),".x3d",4) == 0)
    return X3D;
  else if (len > 4 && strncmp(&(filename[len-4]),".vtk",4) == 0)
    return VTK;
  else if (len > 5 && strncmp(&(filename[len-5]),".cgns",5) == 0)
    return CGNS;
  else if (len > 4 && strncmp(&(filename[len-4]),".off",4) == 0)
    return OFF;
  else {
    fprintf(stderr,"Unrecognized mesh format\n");
    fprintf(stderr,"Recognized mesh formats for import: MSTK, GMV, ExodusII, NemesisI\n");
    fprintf(stderr,"Recognized mesh formats for export: MSTK, GMV, ExodusII, NemesisI, VTK, STL, OFF, DX\n");
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  char infname[256], outfname[256];
  int len, ok;
  int build_classfn=1, partition=-1, weave=-1, use_geometry=0, parallel_check=0;
  int check_topo=0;
  int num_ghost_layers=0, partmethod=0, timing=0;
  int experimental_skip_side_set_attrs=0;
  int experimental_sparse_set_copy=0;
  int experimental_batched_set_copy=0;
  int experimental_preserve_named_sidesets=0;
  int experimental_sparse_sideset_export=0;
  int experimental_ats_exo_workflow=0;
  int experimental_strict_column_partition=0;
  int experimental_preserve_original_element_map=0;
  char verify_column_set[256] = "";
  char summarize_set[256] = "";
  char repair_column_set[256] = "";
  char repair_sparse_column_set[256] = "";
  char sparse_sideset_source[256] = "";
  MshFmt inmeshfmt, outmeshfmt;
  FILE *fp;

  if (argc < 3) {
    fprintf(stderr,"\n");
    fprintf(stderr,"usage: meshconvert <--timing> <--experimental-skip-side-set-attrs> <--experimental-sparse-set-copy> <--experimental-batched-set-copy> <--experimental-preserve-named-sidesets> <--experimental-sparse-sideset-export> <--experimental-sparse-sideset-source=file> <--experimental-ats-exo-workflow> <--experimental-strict-column-partition> <--experimental-preserve-original-element-map> <--summarize-set=name> <--verify-column-set=name> <--repair-column-set=name> <--experimental-repair-sparse-column-set=name> <--classify=0|n|1|y|2> <--partition=y|1|n|0> <--partition-method=0|1|2> <--parallel-check=y|1|n|0> <--weave=y|1|n|0> <--num-ghost-layers=?> <--check-topo=y|1|n|0> infilename outfilename\n\n");
    fprintf(stderr,"partition-method = 0, METIS\n");
    fprintf(stderr,"                 = 1, ZOLTAN with GRAPH partioning\n");
    fprintf(stderr,"                 = 2, ZOLTAN with RCB partitioning\n");
    fprintf(stderr,"Choose 2 if you want to avoid partitioning models\n");
    fprintf(stderr,"with high aspect ratio along the short directions\n");
    fprintf(stderr,"\n");
    fprintf(stderr,"weave = 0, Do not weave distributed meshes for inter-processor connectivity\n");
    fprintf(stderr,"      = 1, Weave distributed meshes for inter-processor connectivity\n");
    fprintf(stderr,"\n");
    fprintf(stderr,"classify = 0/n, No mesh classification information derived\n");
    fprintf(stderr,"         = 1/y, Mesh entity classification derived from material IDs\n");
    fprintf(stderr,"         = 2, As in 1 with additional inference from boundary geometry\n");
    fprintf(stderr,"CLASSIFICATION: Relationship of mesh entities to geometric model/domain\n");
    fprintf(stderr,"\n");
    fprintf(stderr,"check-topo = 0/n, No checking of topological coonsistency of classification info\n");
    fprintf(stderr,"           = 1/y, Check topological consistency of classification info\n");
    fprintf(stderr,"\n");
    exit(-1);
  }

#ifdef MSTK_HAVE_MPI
  MPI_Init(&argc,&argv);
#endif


  MSTK_Init();


  int rank=0, numprocs=1;
#ifdef MSTK_HAVE_MPI

  MSTK_Comm comm = MPI_COMM_WORLD;

  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&numprocs);

#else
  MSTK_Comm comm = NULL;
#endif

  if (rank == 0) {
    fprintf(stderr,"\nApp to convert unstructured meshes between formats\n");
    fprintf(stderr,"Contact: Rao Garimella (rao@lanl.gov)\n\n");
  }

  double total_t0 = meshconvert_wtime();
  double parse_t0 = meshconvert_wtime();

  if (argc > 3) {
    int i;
    for (i = 1; i < argc-2; i++) {
      if (strncmp(argv[i],"--timing",8) == 0) {
        timing = 1;
        setenv("MSTK_MESHCONVERT_TIMING", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-skip-side-set-attrs",34) == 0) {
        experimental_skip_side_set_attrs = 1;
        setenv("MSTK_SKIP_SIDE_SET_ATTR_COPY", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-sparse-set-copy",30) == 0) {
        experimental_sparse_set_copy = 1;
        setenv("MSTK_SPARSE_SET_COPY", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-batched-set-copy",31) == 0) {
        experimental_batched_set_copy = 1;
        setenv("MSTK_BATCHED_SET_COPY", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-preserve-named-sidesets",38) == 0) {
        experimental_preserve_named_sidesets = 1;
        setenv("MSTK_PRESERVE_NAMED_SIDESETS", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-sparse-sideset-export",37) == 0) {
        experimental_sparse_sideset_export = 1;
        setenv("MSTK_SPARSE_SIDESET_EXPORT", "1", 1);
        setenv("MSTK_SKIP_SIDE_SET_ATTR_COPY", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-sparse-sideset-source=",37) == 0) {
        experimental_sparse_sideset_export = 1;
        strncpy(sparse_sideset_source, argv[i]+37,
                sizeof(sparse_sideset_source)-1);
        sparse_sideset_source[sizeof(sparse_sideset_source)-1] = '\0';
        setenv("MSTK_SPARSE_SIDESET_EXPORT", "1", 1);
        setenv("MSTK_SKIP_SIDE_SET_ATTR_COPY", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-ats-exo-workflow",31) == 0) {
        experimental_ats_exo_workflow = 1;
      }
      else if (strncmp(argv[i],"--experimental-strict-column-partition",38) == 0) {
        experimental_strict_column_partition = 1;
        setenv("MSTK_STRICT_COLUMN_PARTITION", "1", 1);
      }
      else if (strncmp(argv[i],"--experimental-preserve-original-element-map",44) == 0) {
        experimental_preserve_original_element_map = 1;
      }
      else if (strncmp(argv[i],"--summarize-set=",16) == 0) {
        strncpy(summarize_set, argv[i]+16, sizeof(summarize_set)-1);
        summarize_set[sizeof(summarize_set)-1] = '\0';
      }
      else if (strncmp(argv[i],"--verify-column-set=",20) == 0) {
        strncpy(verify_column_set, argv[i]+20, sizeof(verify_column_set)-1);
        verify_column_set[sizeof(verify_column_set)-1] = '\0';
      }
      else if (strncmp(argv[i],"--repair-column-set=",20) == 0) {
        strncpy(repair_column_set, argv[i]+20, sizeof(repair_column_set)-1);
        repair_column_set[sizeof(repair_column_set)-1] = '\0';
      }
      else if (strncmp(argv[i],"--experimental-repair-sparse-column-set=",40) == 0) {
        strncpy(repair_sparse_column_set, argv[i]+40,
                sizeof(repair_sparse_column_set)-1);
        repair_sparse_column_set[sizeof(repair_sparse_column_set)-1] = '\0';
      }
      else if (strncmp(argv[i],"--classify",10) == 0) {
        if (strncmp(argv[i]+11,"y",1) == 0 ||
            strncmp(argv[i]+11,"1",1) == 0)
          build_classfn = 1;
        else if (strncmp(argv[i]+11,"n",1) == 0 ||
                 strncmp(argv[i]+11,"0",1) == 0)
          build_classfn = 0;
        else if (strncmp(argv[i]+11,"2",1) == 0) {
          build_classfn = 1;
          use_geometry = 1;
        }
        else
          MSTK_Report("meshconvert",
                      "--classify option should be 0, 1, 2, y or n",
                      MSTK_FATAL);          
           
      }
      else if (strncmp(argv[i],"--partition=",12) == 0) {
        if (strncmp(argv[i]+12,"y",1) == 0 ||
            strncmp(argv[i]+12,"1",1) == 0)
          partition = 2;  // new method for partitioning serial file
        else if (strncmp(argv[i]+12,"o",1) == 0 ||
                 strncmp(argv[i]+12,"2",1) == 0) 
          partition = 1;  // old method for partitioning serial file
        else if (strncmp(argv[i]+12,"n",1) == 0 ||
                 strncmp(argv[i]+12,"0",1) == 0) 
          partition = 0;  // no partitioning
        else
          MSTK_Report("meshconvert",
                      "--partition option should be y, n, 1 or 0",
                      MSTK_FATAL);          
      }
      else if (strncmp(argv[i],"--partition-method",18) == 0 ||
               strncmp(argv[i],"--partition_method",18) == 0) {
        sscanf(argv[i]+19,"%d",&partmethod);
      }
      else if (strncmp(argv[i],"--parallel-check",16) == 0 ||
               strncmp(argv[i],"--parallel_check",16) == 0) {
        if (strncmp(argv[i]+17,"y",1) == 0 ||
            strncmp(argv[i]+17,"1",1) == 1)
          parallel_check = 1;
        else if (strncmp(argv[i]+17,"n",13) == 0 ||
                 strncmp(argv[i]+17,"0",13) == 0) 
          parallel_check = 0;
      }
      else if (strncmp(argv[i],"--weave",7) == 0) {
        if (strncmp(argv[i],"--weave=y",11) == 0 ||
            strncmp(argv[i],"--weave=1",11) == 0)
          weave = 1;
        else if (strncmp(argv[i],"--weave=n",11) == 0 ||
                 strncmp(argv[i],"--weave=0",11) == 0) 
          weave = 0;
        else
          MSTK_Report("meshconvert",
                      "--weave option should be y, n, 1 or 0",
                      MSTK_FATAL);          
           
      }
      else if (strncmp(argv[i],"--num_ghost_layers",18) == 0) {
        sscanf(argv[i]+19,"%d",&num_ghost_layers);
      }
      else if (strncmp(argv[i],"--check-topo",12) == 0) {
        if (strncmp(argv[i],"--check-topo=y",14) == 0 ||
            strncmp(argv[i],"--check-topo=1",14) == 0)
          check_topo=1;
      }
      else
        fprintf(stderr,"Unrecognized option...Ignoring\n");
    }
  }
  double parse_t1 = meshconvert_wtime();

  /* what format is the input mesh file in? */
  strcpy(infname,argv[argc-2]);
  inmeshfmt = getFormat(infname);
  if (inmeshfmt == EXODUSII && weave == 1) {
    inmeshfmt = NEMESISI;
  }

  /* what format should the output mesh file be in? */
  strcpy(outfname,argv[argc-1]);
  outmeshfmt = getFormat(outfname);

  if (timing)
    meshconvert_print_time("argument parsing", parse_t0, parse_t1, rank, comm);



  /* Do we have single or multiple input files? */
  int serial_file = 0, parallel_file = 0;
  if ((fp = fopen(infname,"r"))) {
    serial_file = 1;
    fclose(fp);
  }

#ifdef MSTK_HAVE_MPI
  if (numprocs > 1) {
    /* Assume that all parallel files are of the form
       "base.ext.numprocs.rank" or "base.ext.rank"*/

    int filecount = 0;
    int ndigits = (int)(floor(log10(numprocs))+1);
    for (int r = 0; r < numprocs; r++) {
      char tmpfname1[256], tmpfname2[256], tmpfname3[256];
      sprintf(tmpfname1,"%s.%0*d.%0*d",infname,ndigits,numprocs,ndigits,r);
      sprintf(tmpfname2,"%s.%05d",infname,r);
      sprintf(tmpfname3,"%s.%05d",infname,r+1);  // X3D
      if ((fp = fopen(tmpfname1,"r")) || (fp = fopen(tmpfname2,"r")) || (fp = fopen(tmpfname3,"r"))) {
	filecount++;
	fclose(fp);
      }
    }
    parallel_file = (filecount == numprocs) ? 1 : 0;


    if (serial_file && parallel_file) {
      /* If both serial and parallel files are present, warn the user
	 if the defaults don't make sense */
    
      if (partition > 0 && weave > 0) { /* conflicting options were specified */
	MSTK_Report("meshconvert",
		    "Have serial and parallel input files and BOTH partition and weave options were specified. Pick one: --partition=1|2 will partiton the serial file and --weave=1|y will weave the parallel files",
		    MSTK_FATAL);
      } else if ((partition == -1) && (weave == -1)) {
	if (rank == 0)
	  MSTK_Report("meshconvert",
		      "Have serial and parallel input files but NEITHER partition nor weave options were specifeid. Partitioning serial file", MSTK_WARN);
	partition = 1;
      }
      
    } else if (serial_file || parallel_file) {
      if (serial_file && partition == -1)  /* serial file and partition option not set */
	partition = 1;
      else if (parallel_file && weave == -1)  /* parallel files and weave option not set */
	weave = 1;
    } else {
      MSTK_Report("meshconvert",
		  "Did not find input file",MSTK_FATAL);
    }
  }
#endif

  if (timing && rank == 0) {
    fprintf(stderr,
            "[meshconvert][timing] enabled ranks=%d partition=%d partition-method=%d classify=%d serial_file=%d parallel_file=%d weave=%d input=%s output=%s\n",
            numprocs, partition, partmethod, build_classfn, serial_file,
            parallel_file, weave, infname, outfname);
    if (experimental_skip_side_set_attrs)
      fprintf(stderr,
              "[meshconvert][timing] experimental-skip-side-set-attrs enabled\n");
    if (experimental_sparse_set_copy)
      fprintf(stderr,
              "[meshconvert][timing] experimental-sparse-set-copy enabled\n");
    if (experimental_batched_set_copy)
      fprintf(stderr,
              "[meshconvert][timing] experimental-batched-set-copy enabled\n");
    if (experimental_preserve_named_sidesets)
      fprintf(stderr,
              "[meshconvert][timing] experimental-preserve-named-sidesets enabled\n");
    if (experimental_sparse_sideset_export)
      fprintf(stderr,
              "[meshconvert][timing] experimental-sparse-sideset-export enabled\n");
    if (sparse_sideset_source[0] != '\0')
      fprintf(stderr,
              "[meshconvert][timing] experimental-sparse-sideset-source=%s\n",
              sparse_sideset_source);
    if (experimental_ats_exo_workflow) {
      if (meshconvert_enable_unsafe_ats_exo_workflow())
        fprintf(stderr,
                "[meshconvert][timing] experimental-ats-exo-workflow enabled "
                "(unsafe branch explicitly allowed)\n");
      else
        fprintf(stderr,
                "[meshconvert][timing] experimental-ats-exo-workflow requested; "
                "using stock Exodus partition path for .par equivalence\n");
    }
    if (experimental_strict_column_partition)
      fprintf(stderr,
              "[meshconvert][timing] experimental-strict-column-partition enabled\n");
    if (experimental_preserve_original_element_map)
      fprintf(stderr,
              "[meshconvert][timing] experimental-preserve-original-element-map enabled\n");
    if (verify_column_set[0] != '\0')
      fprintf(stderr,
              "[meshconvert][timing] verify-column-set=%s\n",
              verify_column_set);
    if (summarize_set[0] != '\0')
      fprintf(stderr,
              "[meshconvert][timing] summarize-set=%s\n",
              summarize_set);
    if (repair_column_set[0] != '\0')
      fprintf(stderr,
              "[meshconvert][timing] repair-column-set=%s\n",
              repair_column_set);
    if (repair_sparse_column_set[0] != '\0')
      fprintf(stderr,
              "[meshconvert][timing] experimental-repair-sparse-column-set=%s\n",
              repair_sparse_column_set);
  }

  if (sparse_sideset_source[0] != '\0')
    setenv("MSTK_SPARSE_SIDESET_EXPORT_FILE", sparse_sideset_source, 1);
  else if (experimental_sparse_sideset_export && inmeshfmt == EXODUSII)
    setenv("MSTK_SPARSE_SIDESET_EXPORT_FILE", infname, 1);

  /* now read the mesh */

  Mesh_ptr mesh = NULL;
  int opts[5]={0,0,0,0,0};
  
  if (serial_file) {
    if (inmeshfmt == EXODUSII) {
      /* Exodus II read is special - we have a way of doing a faster reading
	 the serial file and partitioning */
      
      if (rank == 0)
	fprintf(stderr,"Importing mesh from ExodusII file...");

      if (experimental_ats_exo_workflow &&
          meshconvert_enable_unsafe_ats_exo_workflow() &&
          partition > 0) {
#ifdef MSTK_HAVE_MPI
        Mesh_ptr serial_mesh = NULL;
        int dim = 0;
        int ring = 1;
        int with_attr = 1;
        int del_inmesh = 1;

        if (rank == 0) {
          serial_mesh = MESH_New(F1);
          double import_t0 = meshconvert_wtime();
          ok = MESH_ImportFromFile(serial_mesh,infname,"exodusii",NULL,comm);
          double import_t1 = meshconvert_wtime();
          if (timing)
            meshconvert_print_time("ATS MESH_ImportFromExodusII",
                                   import_t0, import_t1, rank, NULL);
          if (!ok)
            MSTK_Report("meshconvert",
                        "ATS workflow Exodus import failed", MSTK_FATAL);

          double renum_t0 = meshconvert_wtime();
          MESH_Renumber(serial_mesh,0,MALLTYPE);
          double renum_t1 = meshconvert_wtime();
          if (timing)
            meshconvert_print_time("ATS MESH_Renumber local IDs",
                                   renum_t0, renum_t1, rank, NULL);

          dim = MESH_Num_Regions(serial_mesh) ? 3 : 2;
        }
        MPI_Bcast(&dim, 1, MPI_INT, 0, comm);

        double dist_t0 = meshconvert_wtime();
        ok = MSTK_Mesh_Distribute(serial_mesh, &mesh, &dim, ring, with_attr,
                                  partmethod, del_inmesh, comm);
        double dist_t1 = meshconvert_wtime();
        if (timing) {
          meshconvert_print_time("ATS MSTK_Mesh_Distribute",
                                 dist_t0, dist_t1, rank, comm);
          meshconvert_print_memory("after ATS MSTK_Mesh_Distribute",
                                   rank, comm);
        }
        if (!ok)
          MSTK_Report("meshconvert",
                      "ATS workflow MSTK_Mesh_Distribute failed",
                      MSTK_FATAL);

        if (experimental_sparse_sideset_export) {
          MAttrib_ptr orig_gid_att =
            MAttrib_New(mesh,MSTK_ATS_ORIG_ELEM_GID_ATT,INT,MREGION);
          MRegion_ptr mr;
          int idx = 0;

          if (!orig_gid_att)
            MSTK_Report("meshconvert",
                        "Could not create original Exodus element id attribute",
                        MSTK_FATAL);

          while ((mr = MESH_Next_Region(mesh,&idx)))
            MEnt_Set_AttVal(mr,orig_gid_att,MR_GlobalID(mr),0.0,NULL);

          setenv("MSTK_SPARSE_SIDESET_OWNER_ATTR",
                 MSTK_ATS_ORIG_ELEM_GID_ATT,1);
          if (experimental_preserve_original_element_map)
            setenv("MSTK_EXODUS_ELEM_MAP_ATTR",
                   MSTK_ATS_ORIG_ELEM_GID_ATT,1);
        }

        double gid_t0 = meshconvert_wtime();
        ok = MESH_Renumber_GlobalIDs(mesh,MALLTYPE,0,NULL,comm);
        double gid_t1 = meshconvert_wtime();
        if (timing)
          meshconvert_print_time("ATS MESH_Renumber_GlobalIDs",
                                 gid_t0, gid_t1, rank, comm);
        if (!ok)
          MSTK_Report("meshconvert",
                      "ATS workflow MESH_Renumber_GlobalIDs failed",
                      MSTK_FATAL);
#else
        MSTK_Report("meshconvert",
                    "ATS Exodus workflow partitioning requires MPI",
                    MSTK_FATAL);
#endif
      }
      else {
        opts[0] = (partition > 0) ? 1 : 0;
        opts[1] = (partition > 0) ? partition-1 : 0;
        opts[2] = 1;  /* 1 layer of ghosts */
        opts[3] = partmethod;

        mesh = MESH_New(F1);
        double import_t0 = meshconvert_wtime();
        ok = MESH_ImportFromFile(mesh,infname,"exodusii",opts,comm);
        double import_t1 = meshconvert_wtime();
        if (timing) {
          meshconvert_print_time("MESH_ImportFromFile", import_t0, import_t1,
                                 rank, comm);
          meshconvert_print_memory("after MESH_ImportFromFile", rank, comm);
        }
      }

    } else {

      Mesh_ptr serial_mesh = NULL;


      if (rank == 0) {
	ok = 0;
	switch(inmeshfmt) {
	case MSTK: {
	  serial_mesh = MESH_New(UNKNOWN_REP);
	  fprintf(stderr,"Reading file in MSTK format...");
          double import_t0 = meshconvert_wtime();
	  ok = MESH_InitFromFile(serial_mesh,infname,comm);
          double import_t1 = meshconvert_wtime();
          if (timing)
            meshconvert_print_time("MESH_InitFromFile", import_t0, import_t1,
                                   rank, NULL);
	  fprintf(stderr,"Done\n");
	  break;
	}
	case GMV: {
	  fprintf(stderr,"Importing mesh from GMV file...");
	  serial_mesh = MESH_New(F1);
          double import_t0 = meshconvert_wtime();
	  ok = MESH_ImportFromFile(serial_mesh,infname,"gmv",opts,comm);
          double import_t1 = meshconvert_wtime();
          if (timing)
            meshconvert_print_time("MESH_ImportFromFile", import_t0, import_t1,
                                   rank, NULL);
	  break;
	}
	case X3D: {
	  fprintf(stderr,"Importing mesh from X3D format...");
	  serial_mesh = MESH_New(F1);
          double import_t0 = meshconvert_wtime();
	  ok = MESH_ImportFromFile(serial_mesh,infname,"x3d",opts,comm);
          double import_t1 = meshconvert_wtime();
          if (timing)
            meshconvert_print_time("MESH_ImportFromFile", import_t0, import_t1,
                                   rank, NULL);
	  break;
	}
	case CGNS: case VTK: case AVSUCD: 
	  if (rank == 0)
	    MSTK_Report("meshconvert","Cannot import mesh from this format. ",
			MSTK_FATAL);
	  break;
	default:
	  if (rank == 0)
	    MSTK_Report("meshconvert","Cannot import from unrecognized format. ",
			MSTK_FATAL);
	}

	if (ok)
	  fprintf(stderr,"Done\n");
	else {
	  fprintf(stderr,"Failed\n");
	  exit(-1);
	}
      }

      if (partition > 0) {
	int ring = 1; /* 1 layer of ghost elements */
	int with_attr = 1; /* Do allow exchange of attributes */
	int del_inmesh = 1; /* Delete input mesh after partitioning */

	int dim;
	if (rank == 0)
	  dim = MESH_Num_Regions(serial_mesh) ? 3 : 2;

#ifdef MSTK_HAVE_MPI
	MPI_Bcast(&dim, 1, MPI_INT, 0, comm);
        double dist_t0 = meshconvert_wtime();
	int ok = MSTK_Mesh_Distribute(serial_mesh, &mesh, &dim, ring, with_attr,
				      partmethod, del_inmesh, comm);
        double dist_t1 = meshconvert_wtime();
        if (timing) {
          meshconvert_print_time("MSTK_Mesh_Distribute", dist_t0, dist_t1,
                                 rank, comm);
          meshconvert_print_memory("after MSTK_Mesh_Distribute", rank, comm);
        }
#else
        MSTK_Report("meshconvert",
                    "Request for partitioning in serial run - use mpirun",
                    MSTK_FATAL);
#endif
      } else
	mesh = serial_mesh;
    }

    /* Figure out geometric classification */
    
    if (experimental_ats_exo_workflow && build_classfn && rank == 0 && timing)
      fprintf(stderr,
              "[meshconvert][timing] skipping MESH_BuildClassfn in ATS Exodus workflow mode\n");
    if (build_classfn && !experimental_ats_exo_workflow) {  /* works correctly only for un-partitioned meshes */
      if (rank == 0) {
	fprintf(stderr,"Building classification information....");
	
        double class_t0 = meshconvert_wtime();
	ok = MESH_BuildClassfn(mesh,use_geometry);  
        double class_t1 = meshconvert_wtime();
        if (timing)
          meshconvert_print_time("MESH_BuildClassfn", class_t0, class_t1,
                                 rank, NULL);

	if (ok)
	  fprintf(stderr,"Done\n");
	else {
	  fprintf(stderr,"Failed\n");
	  exit(-1);
	}
      }
    }
    
    /* Check that the imported serial mesh topology is ok */

    if (check_topo) {  /* works correctly only for un-partitioned meshes */
      if (rank == 0) {
	fprintf(stderr,"Checking mesh topology....");

        double topo_t0 = meshconvert_wtime();
	ok = MESH_CheckTopo(mesh);
        double topo_t1 = meshconvert_wtime();
        if (timing)
          meshconvert_print_time("MESH_CheckTopo", topo_t0, topo_t1,
                                 rank, NULL);
	
	if (ok)
	  fprintf(stderr,"Done\n");
	else {
	  fprintf(stderr,"Failed\n");
	  exit(-1);
	}
      }
    }
  } else {  /* parallel files */

    char parfilename[256];
    ok = 0;
    switch(inmeshfmt) {
    case MSTK: {
      if (rank == 0)
	fprintf(stderr,"Reading file in MSTK format...");

      sprintf(parfilename,"%s.%-d.%-d",infname,numprocs,rank);
      mesh = MESH_New(UNKNOWN_REP);
      
      double import_t0 = meshconvert_wtime();
      ok = MESH_InitFromFile(mesh,parfilename,comm);
      double import_t1 = meshconvert_wtime();
      if (timing)
        meshconvert_print_time("MESH_InitFromFile", import_t0, import_t1,
                               rank, comm);
      
      break;
    }
    case GMV: {
      if (rank == 0)
	fprintf(stderr,"Importing mesh from GMV file...");

      sprintf(parfilename,"%s.%05d",infname,rank+1);
      mesh = MESH_New(F1);
      
      double import_t0 = meshconvert_wtime();
      ok = MESH_ImportFromFile(mesh,parfilename,"gmv",opts,comm);
      double import_t1 = meshconvert_wtime();
      if (timing)
        meshconvert_print_time("MESH_ImportFromFile", import_t0, import_t1,
                               rank, comm);

      break;
    }
    case NEMESISI: {
      if (rank == 0)
	fprintf(stderr,"Reading file in Exodus/Nemesis format...");

      sprintf(parfilename,"%s.%-d.%-d",infname,numprocs,rank);
      mesh = MESH_New(UNKNOWN_REP);
      
      double import_t0 = meshconvert_wtime();
      ok = MESH_ImportFromFile(mesh,parfilename,"nemesisi",opts,comm);
      double import_t1 = meshconvert_wtime();
      if (timing)
        meshconvert_print_time("MESH_ImportFromFile", import_t0, import_t1,
                               rank, comm);
      
      break;
    }
    case CGNS: {
      if (rank == 0)
	fprintf(stderr,"Cannot import mesh from CGNS format. ");
      break;
    }
    case VTK: {
      if (rank == 0)
	fprintf(stderr,"Cannot import mesh from VTK format. ");
      break;
    }
    case AVSUCD: {
      if (rank == 0)
	  fprintf(stderr,"Cannot import mesh from AVS format. ");
      break;
    }
    case X3D: {
      if (rank == 0)
	fprintf(stderr,"Importing mesh from X3D format...");

      sprintf(parfilename,"%s.%05d",infname,rank+1);
      mesh = MESH_New(F1);

      double import_t0 = meshconvert_wtime();
      ok = MESH_ImportFromFile(mesh,parfilename,"x3d",opts,comm);
      double import_t1 = meshconvert_wtime();
      if (timing)
        meshconvert_print_time("MESH_ImportFromFile", import_t0, import_t1,
                               rank, comm);

      break;
    }
    default:
      fprintf(stderr,"Cannot import from unrecognized format. ");
    }
    
    if (ok)
      fprintf(stderr,"Done\n");
    else {
      fprintf(stderr,"Failed\n");
      exit(-1);
    }

#ifdef MSTK_HAVE_MPI

    if (weave == 1) {

      int input_type = 0;
      if (inmeshfmt == NEMESISI)
	input_type = 1;
      else if (inmeshfmt == X3D)
	input_type = 2;

      int num_ghost_layers = 1;
      int dim = MESH_Num_Regions(mesh) ? 3 : 2;
      double weave_t0 = meshconvert_wtime();
      MSTK_Weave_DistributedMeshes(mesh, dim, num_ghost_layers, input_type,
				   comm);
      double weave_t1 = meshconvert_wtime();
      if (timing)
        meshconvert_print_time("MSTK_Weave_DistributedMeshes", weave_t0,
                               weave_t1, rank, comm);

    }
    
    if (numprocs > 1 && parallel_check == 1) {

      /* Do a parallel consistency check too */
      /* Not checking the mesh geometry here */
      
      double pcheck_t0 = meshconvert_wtime();
      ok = ok && MESH_Parallel_Check(mesh,comm);
      double pcheck_t1 = meshconvert_wtime();
      if (timing)
        meshconvert_print_time("MESH_Parallel_Check", pcheck_t0, pcheck_t1,
                               rank, comm);
      
      int allok = 0;
      MPI_Reduce(&ok,&allok,1,MPI_INT,MPI_MIN,0,comm);
      
      if (rank == 0 && allok)
	fprintf(stderr,"Parallel checks passed...\n");
    }
#endif

  }  /* if (serial_file) {...} else {...} */

  if (repair_column_set[0] != '\0') {
    double repair_t0 = meshconvert_wtime();
    ok = meshconvert_repair_column_set(mesh, repair_column_set, rank, comm);
    double repair_t1 = meshconvert_wtime();
    if (timing)
      meshconvert_print_time("repair-column-set", repair_t0, repair_t1,
                             rank, comm);
    if (!ok)
      MSTK_Report("meshconvert",
                  "Column-set repair failed", MSTK_FATAL);
  }

  if (repair_sparse_column_set[0] != '\0') {
    const char *repair_source = sparse_sideset_source[0] != '\0' ?
      sparse_sideset_source : infname;
    double repair_t0 = meshconvert_wtime();
    ok = meshconvert_repair_sparse_column_set(mesh, repair_source,
                                              repair_sparse_column_set,
                                              rank, comm);
    double repair_t1 = meshconvert_wtime();
    if (timing)
      meshconvert_print_time("repair-sparse-column-set", repair_t0, repair_t1,
                             rank, comm);
    if (!ok)
      MSTK_Report("meshconvert",
                  "Sparse column-set repair failed", MSTK_FATAL);
  }

  if (summarize_set[0] != '\0') {
    double summarize_t0 = meshconvert_wtime();
    ok = meshconvert_summarize_set(mesh, summarize_set, rank, numprocs, comm);
    double summarize_t1 = meshconvert_wtime();
    if (timing)
      meshconvert_print_time("summarize-set", summarize_t0, summarize_t1,
                             rank, comm);
    if (!ok)
      MSTK_Report("meshconvert",
                  "Set summary failed", MSTK_FATAL);
  }

  if (verify_column_set[0] != '\0') {
    double verify_t0 = meshconvert_wtime();
    ok = meshconvert_verify_column_set(mesh, verify_column_set, rank, comm);
    double verify_t1 = meshconvert_wtime();
    if (timing)
      meshconvert_print_time("verify-column-set", verify_t0, verify_t1,
                             rank, comm);
    if (!ok)
      MSTK_Report("meshconvert",
                  "Column-set verification failed", MSTK_FATAL);
  }

    

  if (outmeshfmt == MSTK) {
    if (rank == 0)
      fprintf(stderr,"Writing mesh to MSTK file...");
    double export_t0 = meshconvert_wtime();
    MESH_WriteToFile(mesh,outfname,MESH_RepType(mesh),comm);
    double export_t1 = meshconvert_wtime();
    if (timing)
      meshconvert_print_time("MESH_WriteToFile", export_t0, export_t1,
                             rank, comm);
    fprintf(stderr,"Done\n");
  }
  else {
    double export_t0 = meshconvert_wtime();
    switch(outmeshfmt) {
    case GMV:
      if (rank == 0)
        fprintf(stderr,"Exporting mesh to GMV format...");
      ok = MESH_ExportToFile(mesh,outfname,"gmv",0,NULL,NULL,comm);
      break;
    case EXODUSII:case NEMESISI:
      if (rank == 0)
        fprintf(stderr,"Exporting mesh to ExodusII/NemesisI format...");
      ok = MESH_ExportToFile(mesh,outfname,"exodusii",0,NULL,NULL,comm);
      break;
    case CGNS:
      if (rank == 0)
        fprintf(stderr,"Cannot export to CGNS format. ");
      break;
    case VTK:
      if (rank == 0)
        fprintf(stderr,"Cannot export to VTK format. ");
      break;
    case AVSUCD:
      if (rank == 0)
        fprintf(stderr,"Cannot export to AVS format. ");
      break;
    case X3D:
      if (rank == 0)
        fprintf(stderr,"Exporting mesh to FLAG/X3D format...");
      ok = MESH_ExportToFile(mesh,outfname,"x3d",0,NULL,NULL,comm);
      break;
    case STL:
      if (rank == 0)
        fprintf(stderr,"Exporting mesh to STL format...");
      ok = MESH_ExportToFile(mesh, outfname,"stl",0,NULL,NULL,comm);
      break;
    case OFF:
      if (rank == 0)
        fprintf(stderr,"Exporting mesh to OFF format...");
      ok = MESH_ExportToFile(mesh, outfname,"off",0,NULL,NULL,comm);
      break;
    case DX:
      if (rank == 0)
        fprintf(stderr,"Exporting mesh to DX format...");
      ok = MESH_ExportToDX(mesh, outfname, 1);
      break;
    default:
      if (rank == 0)
        fprintf(stderr,"Cannot export mesh to unrecognized format. \n");      
    }
    double export_t1 = meshconvert_wtime();
    if (timing) {
      meshconvert_print_time("MESH_ExportToFile", export_t0, export_t1,
                             rank, comm);
      meshconvert_print_memory("after MESH_ExportToFile", rank, comm);
    }

    if (rank == 0) {
      if (ok)
        fprintf(stderr,"Done\n");
      else {
        fprintf(stderr,"Failed\n");
        exit(-1);
      }
    }
  }


  MESH_Delete(mesh);
  if (timing) {
    double total_t1 = meshconvert_wtime();
    meshconvert_print_time("total", total_t0, total_t1, rank, comm);
    meshconvert_print_memory("before MPI_Finalize", rank, comm);
  }
 
#ifdef MSTK_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
