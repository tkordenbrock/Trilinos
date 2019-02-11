C Copyright(C) 2011-2017 National Technology & Engineering Solutions of
C Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
C NTESS, the U.S. Government retains certain rights in this software.
C
C Redistribution and use in source and binary forms, with or without
C modification, are permitted provided that the following conditions are
C met:
C
C * Redistributions of source code must retain the above copyright
C    notice, this list of conditions and the following disclaimer.
C
C * Redistributions in binary form must reproduce the above
C   copyright notice, this list of conditions and the following
C   disclaimer in the documentation and/or other materials provided
C   with the distribution.
C
C * Neither the name of NTESS nor the names of its
C   contributors may be used to endorse or promote products derived
C   from this software without specific prior written permission.
C
C THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
C "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
C LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
C A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
C OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
C SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
C LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
C DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
C THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
C (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
C OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

C=======================================================================
      SUBROUTINE EXFCON (IBLK, NUMELB, NUMLNK, LINK, ICONOD)
C=======================================================================

C   --*** DBMIR1 *** (GEN3D) Fixup element connectivity for reflections
C   --   Written by Greg Sjaardema - revised 02/10/89
C   --   Modified from DBOEB1 Written by Amy Gilkey
C   --
C   --Parameters:
C   --   NUMELB - IN - the number of elements in the block
C   --   NUMLNK - IN - the number of nodes per element
C   --   LINK   - IN - the element connectivity for this block
C   --

      INTEGER NUMELB, NUMLNK
      INTEGER LINK(NUMLNK,*)
      INTEGER ICONOD(*)

      DO 20 NLNK = 1, NUMLNK
         DO 10 NE = 1, NUMELB
            INODE = LINK (NLNK, NE)
            ICONOD(INODE) = IBLK
   10    CONTINUE
   20 CONTINUE
      RETURN
      END
