cmake_minimum_required(VERSION 3.10)

# Включение горячей перезагрузки для компиляторов MSVC, если поддерживается.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("2048_game")

set(CMAKE_CXX_STANDARD 14)

# Укажите путь к pybind11
set(pybind11_DIR "C:/Users/Angilena/AppData/Local/Programs/Python/Python313/Lib/site-packages/pybind11/share/cmake/pybind11")

# Найдите pybind11
find_package(pybind11 REQUIRED)

# Добавьте библиотеку game
add_library(game MODULE game.cpp)

# Свяжите библиотеку с pybind11
target_link_libraries(game PRIVATE pybind11::module)