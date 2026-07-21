/*
 #
 #  File        : test_template_types.cpp
 #                ( C++ source file )
 #
 #  Description : A file that instantiate the CImg and CImgList classes
 #                for common template types (bool, char, ...), to check that
 #                everything compiles fine.
 #                ( http://cimg.eu )
 #
 #  Copyright   : David Tschumperlé
 #                ( http://tschumperle.users.greyc.fr/ )
 #
 #  License     : CeCILL v2.0
 #                ( http://www.cecill.info/licences/Licence_CeCILL_V2-en.html )
 #
 #  This software is governed by the CeCILL  license under French law and
 #  abiding by the rules of distribution of free software.  You can  use,
 #  modify and/ or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and  rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
 #
*/

#include "CImg.h"

namespace cimg_library {
  template struct CImg<bool>;
  template struct CImg<unsigned char>;
  template struct CImg<char>;
  template struct CImg<unsigned short>;
  template struct CImg<short>;
  template struct CImg<unsigned int>;
  template struct CImg<int>;
  template struct CImg<unsigned long>;
  template struct CImg<long>;
  template struct CImg<float>;
  template struct CImg<double>;

  template struct CImgList<bool>;
  template struct CImgList<unsigned char>;
  template struct CImgList<char>;
  template struct CImgList<unsigned short>;
  template struct CImgList<short>;
  template struct CImgList<unsigned int>;
  template struct CImgList<int>;
  template struct CImgList<unsigned long>;
  template struct CImgList<long>;
  template struct CImgList<float>;
  template struct CImgList<double>;
}

// Main procedure
//----------------
int main(int argc, char **argv) {
  cimg_usage("Instantiate the CImg<T> and CImgList<T> classes for common template types T");
  return 0;
}
