package org.diaduck.commands;

import net.dv8tion.jda.api.events.message.MessageReceivedEvent;

public class PingCommand implements Command {
    @Override
    public void execute(MessageReceivedEvent event) {
        event.getChannel().sendMessage("latency of " + event.getJDA().getGatewayPing() + "ms").queue();
    }
}