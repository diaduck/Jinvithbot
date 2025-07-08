package org.diaduck.commands;

import java.util.HashMap;
import java.util.Map;

public class CommandRegistry {
    private static final Map<String, Command> commands = new HashMap<>();

    static {
        registerCommand("ping", new PingCommand());
        registerCommand("hello", new HelloCommand());
        registerCommand("help", new HelpCommand());
        registerCommand("prompt", new PromptCommand());
    }

    public static void registerCommand(String name, Command command) {
        commands.put(name, command);
    }

    public static Command getCommand(String name) {
        return commands.get(name);
    }
}