package org.diaduck.commands;

import net.dv8tion.jda.api.events.message.MessageReceivedEvent;

public class PromptCommand implements Command {
    public void execute(MessageReceivedEvent event) {
        String returnMessage = "";


        event.getChannel().sendMessage(returnMessage).queue();
    }
}

