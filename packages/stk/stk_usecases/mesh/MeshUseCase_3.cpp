/*------------------------------------------------------------------------*/
/*                 Copyright 2010 Sandia Corporation.                     */
/*  Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive   */
/*  license for use of this work by or on behalf of the U.S. Government.  */
/*  Export of this program may require a license from the                 */
/*  United States Government.                                             */
/*------------------------------------------------------------------------*/

/**
 * @author H. Carter Edwards  <hcedwar@sandia.gov>
 * @date   June 2008
 */

#include <mesh/MeshUseCase_3.hpp>
//
//----------------------------------------------------------------------

#include <iostream>

#include <Shards_BasicTopologies.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>

#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/Stencils.hpp>
#include <stk_mesh/base/TopologyDimensions.hpp>

//----------------------------------------------------------------------

namespace stk {
namespace mesh {
namespace use_cases {

typedef shards::Hexahedron<8>          Hex8;
typedef shards::Wedge<6>               Wedge6;
typedef shards::Tetrahedron<4>         Tet4;
typedef shards::Pyramid<5>             Pyramid4;
typedef shards::ShellQuadrilateral<4>  ShellQuad4;
typedef shards::ShellTriangle<3>       ShellTriangle3;

UseCase_3_Mesh::UseCase_3_Mesh( stk::ParallelMachine comm, bool doCommit ) :
  m_spatial_dimension(3)
  , m_fem_metaData( m_spatial_dimension )
  , m_bulkData( m_fem_metaData , comm )
  , m_block_hex(        declare_part<Hex8>( m_fem_metaData, "block_1"))
  , m_block_wedge(      declare_part<Wedge6>( m_fem_metaData, "block_2"))
  , m_block_tet(        declare_part<Tet4>( m_fem_metaData,"block_3"))
  , m_block_pyramid(    declare_part<Pyramid4>( m_fem_metaData, "block_4"))
  , m_block_quad_shell( declare_part<ShellQuad4>( m_fem_metaData, "block_5"))
  , m_block_tri_shell(  declare_part<ShellTriangle3>( m_fem_metaData, "block_6"))
  , m_elem_rank( stk::topology::ELEMENT_RANK )
  , m_node_rank( topology::NODE_RANK )
  , m_coordinates_field( m_fem_metaData.declare_field< VectorFieldType >(stk::topology::NODE_RANK, "coordinates" ))
  , m_centroid_field(    m_fem_metaData.declare_field< VectorFieldType >(stk::topology::ELEMENT_RANK, "centroid" ))
  , m_temperature_field( m_fem_metaData.declare_field< ScalarFieldType >(stk::topology::NODE_RANK, "temperature" ))
  , m_volume_field( m_fem_metaData.declare_field< ScalarFieldType >(stk::topology::ELEMENT_RANK, "volume" ))
{
  // Define where fields exist on the mesh:
  Part & universal = m_fem_metaData.universal_part();

  put_field( m_coordinates_field , universal );
  put_field( m_centroid_field , universal );
  put_field( m_temperature_field, universal );
  put_field( m_volume_field, m_block_hex );
  put_field( m_volume_field, m_block_wedge );
  put_field( m_volume_field, m_block_tet );
  put_field( m_volume_field, m_block_pyramid );

  if (doCommit)
    m_fem_metaData.commit();
}

UseCase_3_Mesh::~UseCase_3_Mesh()
{ }

//------------------------------------------------------------------------------
// Use case specific mesh generation data:

enum { SpatialDim = 3 };
enum { node_count = 21 };
enum { number_hex = 3 };
enum { number_wedge = 3 };
enum { number_tetra = 3 };
enum { number_pyramid = 2 };
enum { number_shell_quad = 3 };
enum { number_shell_tri = 3 };

namespace {

// Hard coded node coordinate data for all the nodes in the entire mesh
static const double node_coord_data[ node_count ][ SpatialDim ] = {
  { 0 , 0 , 0 } , { 1 , 0 , 0 } , { 2 , 0 , 0 } , { 3 , 0 , 0 } ,
  { 0 , 1 , 0 } , { 1 , 1 , 0 } , { 2 , 1 , 0 } , { 3 , 1 , 0 } ,
  { 0 , 2 , 0 } , { 1 , 2 , 0 } ,
  { 0 , 0 , -1 } , { 1 , 0 , -1 } , { 2 , 0 , -1 } , { 3 , 0 , -1 } ,
  { 0 , 1 , -1 } , { 1 , 1 , -1 } , { 2 , 1 , -1 } , { 3 , 1 , -1 } ,
  { 0 , 2 , -1 } , { 1 , 2 , -1 } ,
  { 1 , 1 , -2 } };

// Hard coded hex node ids for all the hex nodes in the entire mesh
static const EntityId hex_node_ids[number_hex][ Hex8::node_count ] = {
  { 1 , 2 , 12 , 11 , 5 , 6 , 16 , 15 } ,
  { 2 , 3 , 13 , 12 , 6 , 7 , 17 , 16 } ,
  { 3 , 4 , 14 , 13 , 7 , 8 , 18 , 17 } };

// Hard coded wedge node ids for all the wedge nodes in the entire mesh
static const EntityId wedge_node_ids[number_wedge][ Wedge6::node_count ] = {
  { 15 , 16 , 19 ,  5 ,  6 ,  9 } ,
  { 10 ,  9 ,  6 , 20 , 19 , 16 } ,
  { 16 , 17 , 20 ,  6 ,  7 , 10 } };

// Hard coded tetra node ids for all the tetra nodes in the entire mesh
static const EntityId tetra_node_ids[number_tetra][ Tet4::node_count ] = {
  { 15 , 19 , 16 , 21 } ,
  { 19 , 20 , 16 , 21 } ,
  { 16 , 20 , 17 , 21 } };

// Hard coded pyramid node ids for all the pyramid nodes in the entire mesh
static const EntityId pyramid_node_ids[number_pyramid][ Pyramid4::node_count ] = {
  { 11 , 15 , 16 , 12 , 21 } ,
  { 12 , 16 , 17 , 13 , 21 } };

// Hard coded shell quad node ids for all the shell quad nodes in the entire mesh
static const EntityId shell_quad_node_ids[number_shell_quad][ ShellQuad4::node_count ]={
  { 9 , 6 , 16 , 19 } ,
  { 6 , 7 , 17 , 16 } ,
  { 7 , 8 , 18 , 17 } };

// Hard coded shell tri node ids for all the shell tri nodes in the entire mesh
static const EntityId shell_tri_node_ids[number_shell_tri][ ShellTriangle3::node_count ] ={
  { 19 , 16 , 21 } ,
  { 16 , 17 , 21 } ,
  { 17 , 13 , 21 } };

}

//------------------------------------------------------------------------------

void UseCase_3_Mesh::populate()
{
  // Populate mesh with all node types

  m_bulkData.modification_begin();

  EntityId curr_elem_id = 1;

  // For each element topology declare elements

  for ( unsigned i = 0 ; i < number_hex ; ++i , ++curr_elem_id ) {
    declare_element( m_bulkData, m_block_hex, curr_elem_id, hex_node_ids[i] );
  }

  for ( unsigned i = 0 ; i < number_wedge ; ++i , ++curr_elem_id ) {
    declare_element( m_bulkData, m_block_wedge, curr_elem_id, wedge_node_ids[i] );
  }

  for ( unsigned i = 0 ; i < number_tetra ; ++i , ++curr_elem_id ) {
    declare_element( m_bulkData, m_block_tet, curr_elem_id, tetra_node_ids[i] );
  }

  for ( unsigned i = 0 ; i < number_pyramid ; ++i , ++curr_elem_id ) {
    declare_element( m_bulkData, m_block_pyramid, curr_elem_id, pyramid_node_ids[i] );
  }

  for ( unsigned i = 0 ; i < number_shell_quad ; ++i , ++curr_elem_id ) {
    declare_element( m_bulkData, m_block_quad_shell, curr_elem_id, shell_quad_node_ids[i]);
  }

  for ( unsigned i = 0 ; i < number_shell_tri ; ++i , ++curr_elem_id ) {
    declare_element( m_bulkData, m_block_tri_shell, curr_elem_id, shell_tri_node_ids[i] );
  }

  // For all nodes assign nodal coordinates
  for ( unsigned i = 0 ; i < node_count ; ++i ) {
    Entity const node = m_bulkData.get_entity( m_node_rank , i + 1 );
    double * const coord = stk::mesh::field_data( m_coordinates_field , node );
    coord[0] = node_coord_data[i][0] ;
    coord[1] = node_coord_data[i][1] ;
    coord[2] = node_coord_data[i][2] ;
  }

  m_bulkData.modification_end();
  // No parallel stuff for now
}

// Verify mesh for 6 different parts
bool verifyMesh( const UseCase_3_Mesh & mesh )
{
  bool result = true;

  const BulkData & bulkData = mesh.m_bulkData ;

  std::vector<Bucket *> element_buckets = bulkData.buckets( mesh.m_elem_rank );

  // Create a pair containing Part and matching node_count
  typedef std::pair<Part*, unsigned> PartNodeCountPair;
  std::vector<PartNodeCountPair> part_and_node_counts;
  part_and_node_counts.push_back(PartNodeCountPair(&mesh.m_block_hex, Hex8::node_count));
  part_and_node_counts.push_back(PartNodeCountPair(&mesh.m_block_wedge, Wedge6::node_count));
  part_and_node_counts.push_back(PartNodeCountPair(&mesh.m_block_tet, Tet4::node_count));
  part_and_node_counts.push_back(PartNodeCountPair(&mesh.m_block_pyramid, Pyramid4::node_count));
  part_and_node_counts.push_back(PartNodeCountPair(&mesh.m_block_quad_shell, ShellQuad4::node_count));
  part_and_node_counts.push_back(PartNodeCountPair(&mesh.m_block_tri_shell, ShellTriangle3::node_count));

  // Check that all the nodes were allocated.
  for ( unsigned i = 0 ; i < node_count ; ++i ) {
    Entity const node = bulkData.get_entity( mesh.m_node_rank , i + 1 );
    if ( !bulkData.is_valid(node) ) {
      std::cerr << "Error!  Invalid null pointer for node returned from "
        << "bulkData.get_entity( m_node_rank, " << i+1 << " ) " << std::endl;
      result = false;
    }
  }

  return result;
}

} //namespace use_cases
} //namespace mesh
} //namespace stk
