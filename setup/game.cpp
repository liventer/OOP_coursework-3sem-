#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <iostream>
#include <numeric>  

namespace py = pybind11;

class Game {

private:
    int board_dim;
    int state_size;
    int action_size;
    double negative_reward;
    std::string reward_mode;
    double cell_move_penalty;
    std::vector<std::vector<double>> game_board;
    double score;
    double reward;
    double current_cell_move_penalty;
    bool done;
    int steps;
    std::vector<double> rewards_list;
    std::vector<double> scores_list;
    double step_penalty;
    std::vector<std::tuple<int, const double*, std::vector<std::vector<double>>, std::vector<std::vector<double>>, double, double>> history;
    std::mt19937 rng;
    bool moved;

    std::vector<std::vector<double>> shift(const std::vector<std::vector<double>>& board) {
        std::vector<std::vector<double>> shifted_board(board_dim, std::vector<double>(board_dim, 0.0));
        for (int i = 0; i < board_dim; ++i) {
            std::vector<double> shifted(board_dim, 0.0);
            int idx = 0;
            for (int j = 0; j < board_dim; ++j) {
                if (board[i][j] != 0.0) {
                    shifted[idx] = board[i][j];
                    if (j != idx) {
                        current_cell_move_penalty += cell_move_penalty * board[i][j];
                    }
                    ++idx;
                }
            }
            shifted_board[i] = shifted;
        }
        return shifted_board;
    }

    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& board) {
        std::vector<std::vector<double>> transposed_board(board_dim, std::vector<double>(board_dim, 0.0));
        for (int i = 0; i < board_dim; ++i) {
            for (int j = 0; j < board_dim; ++j) {
                transposed_board[j][i] = board[i][j];
            }
        }
        return transposed_board;
    }

    std::vector<std::vector<double>> flip_horizontal(const std::vector<std::vector<double>>& board) {
        std::vector<std::vector<double>> flipped_board(board_dim, std::vector<double>(board_dim, 0.0));
        for (int i = 0; i < board_dim; ++i) {
            for (int j = 0; j < board_dim; ++j) {
                flipped_board[i][j] = board[i][board_dim - 1 - j];
            }
        }
        return flipped_board;
    }

    std::vector<std::vector<double>> calc_board(const std::vector<std::vector<double>>& board) {
        reward = 0.0;
        current_cell_move_penalty = 0.0;

        std::vector<std::vector<double>> shifted_board = shift(board);
        std::vector<std::vector<double>> merged_board(board_dim, std::vector<double>(board_dim, 0.0));
        for (int i = 0; i < board_dim; ++i) {
            for (int j = 0; j < board_dim - 1; ++j) {
                if (shifted_board[i][j] != 0.0 && shifted_board[i][j] == shifted_board[i][j + 1]) {
                    shifted_board[i][j] *= 2.0;
                    shifted_board[i][j + 1] = 0.0;
                    if (reward_mode == "log2") {
                        reward += std::log2(shifted_board[i][j]);
                    }
                    else {
                        reward += shifted_board[i][j];
                    }
                }
            }
            merged_board[i] = shifted_board[i];
        }
        merged_board = shift(merged_board);
        return merged_board;
    }

    std::vector<std::vector<double>> process_action(int action, const std::vector<std::vector<double>>& board) {
        std::vector<std::vector<double>> temp_board = board;

        if (action == 0) { // ACTION_UP
            temp_board = transpose(calc_board(transpose(temp_board)));
        }
        else if (action == 1) { // ACTION_DOWN
            temp_board = transpose(flip_horizontal(calc_board(flip_horizontal(transpose(temp_board)))));
        }
        else if (action == 2) { // ACTION_LEFT
            temp_board = calc_board(temp_board);
        }
        else if (action == 3) { // ACTION_RIGHT
            temp_board = flip_horizontal(calc_board(flip_horizontal(temp_board)));
        }

        return temp_board;
    }

public:
    Game(int size = 4, int seed = 42, double negative_reward = -10.0, std::string reward_mode = "log2", double cell_move_penalty = 0.1)
        : board_dim(size), state_size(size* size), action_size(4), negative_reward(negative_reward), reward_mode(reward_mode), cell_move_penalty(cell_move_penalty),
        score(0.0), reward(0.0), current_cell_move_penalty(0.0), done(false), steps(0), moved(false), step_penalty(0.0), rng(seed) {}

    void reset(int init_fields = 2, double step_penalty = 0.0) {
        game_board = std::vector<std::vector<double>>(board_dim, std::vector<double>(board_dim, 0.0));

        for (int i = 0; i < init_fields; ++i) {
            fill_random_empty_cell();
        }

        score = std::accumulate(game_board.begin(), game_board.end(), 0.0, [](double sum, const std::vector<double>& row) {
            return sum + std::accumulate(row.begin(), row.end(), 0.0);
            });
        reward = 0.0;
        current_cell_move_penalty = 0.0;
        done = false;
        steps = 0;
        rewards_list.clear();
        scores_list.clear();
        step_penalty = step_penalty;
        history.clear();

        history.push_back({
            -1,
            nullptr,
            game_board,
            std::vector<std::vector<double>>(),
            score,
            reward
            });
    }

    py::array_t<double> current_state() {
        std::vector<double> state;
        for (const auto& row : game_board) {
            state.insert(state.end(), row.begin(), row.end());
        }
        return py::array_t<double>(state.size(), state.data());
    }

    py::tuple step(int action, const py::array_t<double>& action_values) {
        std::vector<std::vector<double>> old_board = game_board;
        std::vector<std::vector<double>> temp_board = process_action(action, game_board);

        if (game_board != temp_board) {
            game_board = temp_board;
            fill_random_empty_cell();
            reward = reward - current_cell_move_penalty;
            score = std::accumulate(game_board.begin(), game_board.end(), 0.0, [](double sum, const std::vector<double>& row) {
                return sum + std::accumulate(row.begin(), row.end(), 0.0);
                });
            done = check_is_done();
            moved = true;
        }
        else {
            reward = negative_reward;
            moved = false;
        }
        steps += 1;
        rewards_list.push_back(reward);

        history.push_back({
            action,
            action_values.data(),
            old_board,
            game_board,
            score,
            reward
            });

        return py::make_tuple(game_board, reward, done);
    }

    bool check_is_done() {
        return check_is_done(game_board);
    }

    bool check_is_done(const std::vector<std::vector<double>>& board) {
        for (const auto& row : board) {
            if (std::find(row.begin(), row.end(), 0.0) != row.end()) {
                return false;
            }
        }

        for (const auto& row : board) {
            for (size_t i = 0; i < row.size() - 1; ++i) {
                if (row[i] == row[i + 1]) {
                    return false;
                }
            }
        }

        for (size_t i = 0; i < board.size(); ++i) {
            for (size_t j = 0; j < board[i].size() - 1; ++j) {
                if (board[j][i] == board[j + 1][i]) {
                    return false;
                }
            }
        }

        return true;
    }

    void fill_random_empty_cell() {
        std::vector<std::pair<int, int>> empty_cells;
        for (int i = 0; i < board_dim; ++i) {
            for (int j = 0; j < board_dim; ++j) {
                if (game_board[i][j] == 0.0) {
                    empty_cells.emplace_back(i, j);
                }
            }
        }

        if (empty_cells.empty()) {
            return;
        }

        std::uniform_int_distribution<int> dist(0, empty_cells.size() - 1);
        int index = dist(rng);
        int x = empty_cells[index].first;
        int y = empty_cells[index].second;

        game_board[x][y] = (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.9) ? 2.0 : 4.0;
    }

    std::vector<std::tuple<int, py::array_t<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, double, double>> get_history() const {
        std::vector<std::tuple<int, py::array_t<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, double, double>> history_copy;
        for (const auto& entry : history) {
            const double* action_values_ptr = std::get<1>(entry);
            py::array_t<double> action_values(action_size, action_values_ptr);

            history_copy.emplace_back(
                std::get<0>(entry),
                action_values,
                std::get<2>(entry),
                std::get<3>(entry),
                std::get<4>(entry),
                std::get<5>(entry)
            );
        }
        return history_copy;
    }

    double get_score() const {
        return score;
    }

    double get_reward() const {
        return reward;
    }

    double get_negative_reward() const {
        return negative_reward;
    }

    bool get_done() const {
        return done;
    }

    bool get_moved() const {
        return moved;
    }

    int get_steps() const {
        return steps;
    }

    void set_moved(bool value) {
        moved = value;
    }

    std::string get_reward_mode() const {
        return reward_mode;
    }

    void draw_board(const std::vector<std::vector<double>>& board, const std::string& title) {
        py::module plt = py::module::import("matplotlib.pyplot");
        py::dict cell_colors = py::dict(
            py::arg("0") = "#FFFFFF",
            py::arg("2") = "#EEE4DA",
            py::arg("4") = "#ECE0C8",
            py::arg("8") = "#ECB280",
            py::arg("16") = "#EC8D53",
            py::arg("32") = "#F57C5F",
            py::arg("64") = "#E95937",
            py::arg("128") = "#F3D96B",
            py::arg("256") = "#F2D04A",
            py::arg("512") = "#E5BF2E",
            py::arg("1024") = "#E2B814",
            py::arg("2048") = "#EBC502",
            py::arg("4096") = "#00A2D8",
            py::arg("8192") = "#9ED682"
        );

        int ncols = board.size();
        int nrows = board.size();

        plt.attr("figure")(py::arg("figsize") = py::make_tuple(3, 3));
        plt.attr("suptitle")(title);
        py::list axes;
        for (int r = 0; r < nrows; ++r) {
            for (int c = 1; c <= ncols; ++c) {
                axes.append(plt.attr("subplot")(nrows, ncols, r * ncols + c));
            }
        }

        std::vector<double> v;
        for (const auto& row : board) {
            v.insert(v.end(), row.begin(), row.end());
        }

        for (size_t i = 0; i < axes.size(); ++i) {
            py::object ax = axes[i];
            ax.attr("text")(0.5, 0.5, std::to_string(static_cast<int>(v[i])),
                py::arg("horizontalalignment") = "center",
                py::arg("verticalalignment") = "center");

            // Используем py::str для доступа к элементам словаря
            ax.attr("set_facecolor")(cell_colors[py::str(std::to_string(static_cast<int>(v[i])))]);
        }

        // Убираем метки осей
        for (const auto& ax : axes) {
            ax.attr("set_xticks")(py::list());
            ax.attr("set_yticks")(py::list());
        }

        plt.attr("show")();
    }
};

PYBIND11_MODULE(game, m) {
    py::class_<Game>(m, "Game")
        .def(py::init<int, int, double, std::string, double>(),
            py::arg("size") = 4,
            py::arg("seed") = 42,
            py::arg("negative_reward") = -10.0,
            py::arg("reward_mode") = "log2",
            py::arg("cell_move_penalty") = 0.1)
        .def("reset", &Game::reset,
            py::arg("init_fields") = 2,
            py::arg("step_penalty") = 0.0)
        .def("current_state", &Game::current_state)
        .def("step", &Game::step)
        .def("check_is_done", py::overload_cast<const std::vector<std::vector<double>>&>(&Game::check_is_done))
        .def("check_is_done", py::overload_cast<>(&Game::check_is_done))
        .def("fill_random_empty_cell", &Game::fill_random_empty_cell)
        .def("get_score", &Game::get_score)
        .def("get_reward", &Game::get_reward)
        .def("get_negative_reward", &Game::get_negative_reward)
        .def("get_done", &Game::get_done)
        .def("get_moved", &Game::get_moved)
        .def("get_steps", &Game::get_steps)
        .def("set_moved", &Game::set_moved)
        .def("get_reward_mode", &Game::get_reward_mode)
        .def("get_history", &Game::get_history)
        .def("draw_board", &Game::draw_board, py::arg("board"), py::arg("title") = "Current game");
}