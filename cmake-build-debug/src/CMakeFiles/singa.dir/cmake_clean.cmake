file(REMOVE_RECURSE
  "../lib/libsinga.pdb"
  "../lib/libsinga.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/singa.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
