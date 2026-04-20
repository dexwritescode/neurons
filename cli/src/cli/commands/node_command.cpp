#include "node_command.h"
#include <iostream>
#include <iomanip>

namespace neurons::cli {

NodeCommand::NodeCommand(NeuronsConfig* config) : config_(config) {
    command_name_ = "node";
    description_  = "Manage Neurons service nodes";
}

void NodeCommand::setup_command(CLI::App& app) {
    auto* cmd = app.add_subcommand(command_name_, description_);

    auto* list = cmd->add_subcommand("list", "List configured nodes");
    list->callback([this]() { subcommand_ = "list"; });

    auto* add = cmd->add_subcommand("add", "Add a remote node");
    add->add_option("name", add_name_, "Node name")->required();
    add->add_option("host", add_host_, "Hostname or IP")->required();
    add->add_option("--port,-p", add_port_, "gRPC port (default: 50051)");
    add->callback([this]() { subcommand_ = "add"; });

    auto* remove = cmd->add_subcommand("remove", "Remove a node by name");
    remove->add_option("name", target_id_, "Node name")->required();
    remove->callback([this]() { subcommand_ = "remove"; });

    auto* use = cmd->add_subcommand("use", "Set the active node");
    use->add_option("name", target_id_, "Node name (or 'local')")->required();
    use->callback([this]() { subcommand_ = "use"; });

    auto* status = cmd->add_subcommand("status", "Show active node");
    status->callback([this]() { subcommand_ = "status"; });

    cmd->require_subcommand(1);
    cmd->callback([this]() { std::exit(execute()); });
    (void)list; (void)remove; (void)use; (void)status;
}

int NodeCommand::execute() {
    if (subcommand_ == "list") {
        const auto& nodes = config_->nodes();
        if (nodes.empty()) {
            std::cout << "No nodes configured. Local inference is always available.\n";
            return 0;
        }
        const auto& active = config_->activeNodeId();
        std::cout << std::left
                  << std::setw(3)  << " "
                  << std::setw(20) << "NAME"
                  << std::setw(30) << "HOST"
                  << "PORT\n";
        std::cout << std::string(55, '-') << "\n";
        for (const auto& n : nodes) {
            bool isActive = (n.id == active || n.name == active);
            std::cout << std::setw(3)  << (isActive ? "*" : " ")
                      << std::setw(20) << n.name
                      << std::setw(30) << n.host
                      << n.port << "\n";
        }
        return 0;
    }

    if (subcommand_ == "add") {
        NodeConfig n;
        n.id   = add_name_; // use name as id for simplicity
        n.name = add_name_;
        n.host = add_host_;
        n.port = add_port_;
        config_->addNode(n);
        if (!config_->save()) return 1;
        std::cout << "Added node '" << add_name_ << "' (" << add_host_ << ":" << add_port_ << ")\n";
        return 0;
    }

    if (subcommand_ == "remove") {
        if (!config_->removeNode(target_id_)) {
            std::cerr << "Node '" << target_id_ << "' not found\n";
            return 1;
        }
        if (!config_->save()) return 1;
        std::cout << "Removed node '" << target_id_ << "'\n";
        return 0;
    }

    if (subcommand_ == "use") {
        if (target_id_ == "local") {
            config_->setActiveNodeId("");
            if (!config_->save()) return 1;
            std::cout << "Using local node\n";
            return 0;
        }
        const auto node = config_->findNode(target_id_);
        if (!node) {
            std::cerr << "Node '" << target_id_ << "' not found. Use 'neurons node list' to see nodes.\n";
            return 1;
        }
        config_->setActiveNodeId(node->id);
        if (!config_->save()) return 1;
        std::cout << "Active node: " << node->name << " (" << node->host << ":" << node->port << ")\n";
        return 0;
    }

    if (subcommand_ == "status") {
        const auto& id = config_->activeNodeId();
        if (id.empty()) {
            std::cout << "Active node: local\n";
            return 0;
        }
        const auto node = config_->findNode(id);
        if (node) {
            std::cout << "Active node: " << node->name << " (" << node->host << ":" << node->port << ")\n";
        } else {
            std::cout << "Active node: " << id << " (not found in config)\n";
        }
        return 0;
    }

    return 1;
}

} // namespace neurons::cli
