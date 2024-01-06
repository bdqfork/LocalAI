package openai

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
	"github.com/valyala/fasthttp"

	"github.com/go-skynet/LocalAI/api/backend"
	config "github.com/go-skynet/LocalAI/api/config"
	"github.com/go-skynet/LocalAI/api/options"
	"github.com/go-skynet/LocalAI/api/schema"
	"github.com/go-skynet/LocalAI/pkg/grammar"
	model "github.com/go-skynet/LocalAI/pkg/model"
	"github.com/go-skynet/LocalAI/pkg/utils"
)

type FunctionCall struct {
	Name       string `json:"name,omitempty"`
	Function   string `json:"function,omitempty"`
	Arguments  any    `json:"arguments,omitempty"`
	Parameters any    `json:"parameters,omitempty"`
}

func ChatEndpoint(cm *config.ConfigLoader, o *options.Option) func(c *fiber.Ctx) error {
	emptyMessage := ""
	id := uuid.New().String()
	created := int(time.Now().Unix())

	process := func(s string, req *schema.OpenAIRequest, config *config.Config, loader *model.ModelLoader, responses chan schema.OpenAIResponse) {
		actionStartWord := config.FunctionsConfig.FunctionStartWord
		actionEndWord := config.FunctionsConfig.FunctionEndWord
		actionIgnoreWords := config.FunctionsConfig.FunctionIgnoreWords

		useAction := len(actionStartWord) > 0 && len(actionEndWord) > 0

		initialMessage := schema.OpenAIResponse{
			ID:      id,
			Created: created,
			Model:   req.Model, // we have to return what the user sent here, due to OpenAI spec.
			Choices: []schema.Choice{{Delta: &schema.Message{Role: "assistant", Content: &emptyMessage}}},
			Object:  "chat.completion.chunk",
		}
		responses <- initialMessage

		buffer := ""
		actionBuffer := ""

		isActionStart := false
		isFirstNotEmptyTokenSent := false

		ComputeChoices(req, s, config, o, loader, func(s string, c *[]schema.Choice) {}, func(s string, usage backend.TokenUsage) bool {
			if !isFirstNotEmptyTokenSent {
				if strings.TrimSpace(s) == "" {
					return true
				}
				isFirstNotEmptyTokenSent = true
			}
			buffer += s
			log.Debug().Msgf("Got output: %+v, buffer: %+v, actionBuffer: %+v", s, buffer, actionBuffer)

			if useAction && isActionStart {
				// TODO: 重构
				if len(actionIgnoreWords) > 0 {
					if strings.HasPrefix(actionIgnoreWords[0], buffer) {
						if actionIgnoreWords[0] == buffer {
							buffer = ""
						}
						return true
					}
				}

				if !strings.HasPrefix(actionEndWord, buffer) {
					actionBuffer += buffer
					buffer = ""
					return true
				}

				log.Debug().Msgf("Match FunctionEndWord prefix: %+v, buffer: %v", actionStartWord, buffer)

				if actionEndWord != buffer {
					return true
				}

				isActionStart = false

				functionCall := &FunctionCall{}
				_ = json.Unmarshal([]byte(actionBuffer), functionCall)
				if functionCall.Arguments == nil {
					functionCall.Arguments = functionCall.Parameters
				}
				if _, ok := functionCall.Arguments.(string); functionCall.Arguments != nil && !ok {
					data, _ := json.Marshal(functionCall.Arguments)
					functionCall.Arguments = string(data)
				}

				actionBuffer = ""
				buffer = ""

				resp := schema.OpenAIResponse{
					ID:      id,
					Created: created,
					Model:   req.Model, // we have to return what the user sent here, due to OpenAI spec.
					Choices: []schema.Choice{{Delta: &schema.Message{FunctionCall: functionCall}, Index: 0}},
					Object:  "chat.completion.chunk",
					Usage: schema.OpenAIUsage{
						PromptTokens:     usage.Prompt,
						CompletionTokens: usage.Completion,
						TotalTokens:      usage.Prompt + usage.Completion,
					},
				}
				responses <- resp

				return true
			}

			if useAction && strings.HasPrefix(actionStartWord, buffer) {
				log.Debug().Msgf("Match FunctionStartWord prefix: %+v, buffer: %v", actionStartWord, buffer)
				if actionStartWord == buffer {
					isActionStart = true
					buffer = ""
				}
				return true
			}

			resp := schema.OpenAIResponse{
				ID:      id,
				Created: created,
				Model:   req.Model, // we have to return what the user sent here, due to OpenAI spec.
				Choices: []schema.Choice{{Delta: &schema.Message{Content: buffer}, Index: 0}},
				Object:  "chat.completion.chunk",
				Usage: schema.OpenAIUsage{
					PromptTokens:     usage.Prompt,
					CompletionTokens: usage.Completion,
					TotalTokens:      usage.Prompt + usage.Completion,
				},
			}
			responses <- resp
			buffer = ""
			return true
		})
		close(responses)
	}
	return func(c *fiber.Ctx) error {
		processFunctions := false
		funcs := grammar.Functions{}
		modelFile, input, err := readRequest(c, o, true)
		if err != nil {
			return fmt.Errorf("failed reading parameters from request:%w", err)
		}
		log.Debug().Msgf("Request intput: %+v", input)

		config, input, err := mergeRequestWithConfig(modelFile, input, cm, o.Loader, o.Debug, o.Threads, o.ContextSize, o.F16)
		if err != nil {
			return fmt.Errorf("failed reading parameters from request:%w", err)
		}
		log.Debug().Msgf("Configuration read: %+v", config)

		// Allow the user to set custom actions via config file
		// to be "embedded" in each model
		noActionName := "answer"
		noActionDescription := "use this action to answer without performing any action"

		if config.FunctionsConfig.NoActionFunctionName != "" {
			noActionName = config.FunctionsConfig.NoActionFunctionName
		}
		if config.FunctionsConfig.NoActionDescriptionName != "" {
			noActionDescription = config.FunctionsConfig.NoActionDescriptionName
		}

		if input.ResponseFormat.Type == "json_object" {
			input.Grammar = grammar.JSONBNF
		}

		// process functions if we have any defined or if we have a function call string
		if len(input.Functions) > 0 && config.ShouldUseFunctions() {
			log.Debug().Msgf("Response needs to process functions")

			processFunctions = true

			// Append the no action function
			funcs = append(funcs, input.Functions...)

			if config.MustUseFunctions() || !config.FunctionsConfig.DisableNoAction {
				noActionGrammar := grammar.Function{
					Name:        noActionName,
					Description: noActionDescription,
					Parameters: map[string]interface{}{
						"properties": map[string]interface{}{
							"message": map[string]interface{}{
								"type":        "string",
								"description": "The message to reply the user with",
							}},
					},
				}

				funcs = append(funcs, noActionGrammar)

				// Force picking one of the functions by the request
				if config.FunctionToCall() != "" {
					funcs = funcs.Select(config.FunctionToCall())
				}

				// Update input grammar
				jsStruct := funcs.ToJSONStructure()
				config.Grammar = jsStruct.Grammar("")
			}
		} else if input.JSONFunctionGrammarObject != nil {
			config.Grammar = input.JSONFunctionGrammarObject.Grammar("")
		}

		// functions are not supported in stream mode (yet?)
		toStream := input.Stream
		// toStream := input.Stream && !processFunctions

		log.Debug().Msgf("Parameters: %+v", config)

		var predInput string

		suppressConfigSystemPrompt := false
		mess := []string{}
		for messageIndex, i := range input.Messages {
			var content string
			role := i.Role

			// if function call, we might want to customize the role so we can display better that the "assistant called a json action"
			// if an "assistant_function_call" role is defined, we use it, otherwise we use the role that is passed by in the request
			if i.FunctionCall != nil && i.Role == "assistant" {
				roleFn := "assistant_function_call"
				r := config.Roles[roleFn]
				if r != "" {
					role = roleFn
				}
			}
			r := config.Roles[role]
			contentExists := i.Content != nil && i.StringContent != ""
			// First attempt to populate content via a chat message specific template
			if config.TemplateConfig.ChatMessage != "" {
				chatMessageData := model.ChatMessageTemplateData{
					SystemPrompt: config.SystemPrompt,
					Role:         r,
					RoleName:     role,
					Content:      i.StringContent,
					MessageIndex: messageIndex,
					FunctionCall: i.FunctionCall,
				}
				templatedChatMessage, err := o.Loader.EvaluateTemplateForChatMessage(config.TemplateConfig.ChatMessage, chatMessageData)
				if err != nil {
					log.Error().Msgf("error processing message %+v using template \"%s\": %v. Skipping!", chatMessageData, config.TemplateConfig.ChatMessage, err)
				} else {
					if templatedChatMessage == "" {
						log.Warn().Msgf("template \"%s\" produced blank output for %+v. Skipping!", config.TemplateConfig.ChatMessage, chatMessageData)
						continue // TODO: This continue is here intentionally to skip over the line `mess = append(mess, content)` below, and to prevent the sprintf
					}
					log.Debug().Msgf("templated message for chat: %s", templatedChatMessage)
					content = templatedChatMessage
				}
			}
			// If this model doesn't have such a template, or if that template fails to return a value, template at the message level.
			if content == "" {
				if r != "" {
					if contentExists {
						content = fmt.Sprint(r, i.StringContent)
					}
					if i.FunctionCall != nil {
						j, err := json.Marshal(i.FunctionCall)
						if err == nil {
							if contentExists {
								content += "\n" + fmt.Sprint(r, " ", string(j))
							} else {
								content = fmt.Sprint(r, " ", string(j))
							}
						}
					}
				} else {
					if contentExists {
						content = fmt.Sprint(i.StringContent)
					}
					if i.FunctionCall != nil {
						j, err := json.Marshal(i.FunctionCall)
						if err == nil {
							if contentExists {
								content += "\n" + string(j)
							} else {
								content = string(j)
							}
						}
					}
				}
				// Special Handling: System. We care if it was printed at all, not the r branch, so check seperately
				if contentExists && role == "system" {
					suppressConfigSystemPrompt = true
				}
			}

			mess = append(mess, content)
		}

		predInput = strings.Join(mess, "\n")
		log.Debug().Msgf("Prompt (before templating): %s", predInput)

		if toStream {
			log.Debug().Msgf("Stream request received")
			c.Context().SetContentType("text/event-stream")
			//c.Response().Header.SetContentType(fiber.MIMETextHTMLCharsetUTF8)
			//	c.Set("Content-Type", "text/event-stream")
			c.Set("Cache-Control", "no-cache")
			c.Set("Connection", "keep-alive")
			c.Set("Transfer-Encoding", "chunked")
		}

		templateFile := ""

		// A model can have a "file.bin.tmpl" file associated with a prompt template prefix
		if o.Loader.ExistsInModelPath(fmt.Sprintf("%s.tmpl", config.Model)) {
			templateFile = config.Model
		}

		if config.TemplateConfig.Chat != "" && !processFunctions {
			templateFile = config.TemplateConfig.Chat
		}

		if config.TemplateConfig.Functions != "" && processFunctions {
			templateFile = config.TemplateConfig.Functions
		}

		if templateFile != "" {
			templatedInput, err := o.Loader.EvaluateTemplateForPrompt(model.ChatPromptTemplate, templateFile, model.PromptTemplateData{
				SystemPrompt:         config.SystemPrompt,
				SuppressSystemPrompt: suppressConfigSystemPrompt,
				Input:                predInput,
				Functions:            funcs,
			})
			if err == nil {
				predInput = templatedInput
				log.Debug().Msgf("Template found, input modified to: %s", predInput)
			} else {
				log.Debug().Msgf("Template failed loading: %s", err.Error())
			}
		}

		log.Debug().Msgf("Prompt (after templating): %s", predInput)
		if processFunctions {
			log.Debug().Msgf("Grammar: %+v", config.Grammar)
		}

		if toStream {
			responses := make(chan schema.OpenAIResponse)

			go process(predInput, input, config, o.Loader, responses)

			c.Context().SetBodyStreamWriter(fasthttp.StreamWriter(func(w *bufio.Writer) {

				usage := &schema.OpenAIUsage{}

				hasFunctionCalling := false
				for ev := range responses {
					if ev.Choices[0].Delta.FunctionCall != nil {
						hasFunctionCalling = true
					}

					usage = &ev.Usage // Copy a pointer to the latest usage chunk so that the stop message can reference it
					var buf bytes.Buffer
					enc := json.NewEncoder(&buf)
					enc.Encode(ev)
					log.Debug().Msgf("Sending chunk: %s", buf.String())
					_, err := fmt.Fprintf(w, "data: %v\n", buf.String())
					if err != nil {
						log.Debug().Msgf("Sending chunk failed: %v", err)
						input.Cancel()
						break
					}
					w.Flush()
				}

				finishReason := "stop"
				if hasFunctionCalling {
					finishReason = "function_call"
				}

				resp := &schema.OpenAIResponse{
					ID:      id,
					Created: created,
					Model:   input.Model, // we have to return what the user sent here, due to OpenAI spec.
					Choices: []schema.Choice{
						{
							FinishReason: finishReason,
							Index:        0,
							Delta:        &schema.Message{Content: &emptyMessage},
						}},
					Object: "chat.completion.chunk",
					Usage:  *usage,
				}
				respData, _ := json.Marshal(resp)

				w.WriteString(fmt.Sprintf("data: %s\n\n", respData))
				w.WriteString("data: [DONE]\n\n")
				w.Flush()
			}))
			return nil
		}

		result, tokenUsage, err := ComputeChoices(input, predInput, config, o, o.Loader, func(s string, c *[]schema.Choice) {
			log.Debug().Msgf("Got output: %+v", s)

			// This prevent newlines to break JSON parsing for clients
			s = utils.EscapeNewLines(s)

			// As we have to change the result before processing, we can't stream the answer (yet?)
			ss := parseFunctionCall(s, config.FunctionsConfig)

			if ss != nil {
				*c = append(*c, schema.Choice{
					FinishReason: "function_call",
					Message:      &schema.Message{Role: "assistant", FunctionCall: ss},
				})
				return
			}
			*c = append(*c, schema.Choice{FinishReason: "stop", Index: 0, Message: &schema.Message{Role: "assistant", Content: &s}})
		}, nil)

		if err != nil {
			return err
		}

		resp := &schema.OpenAIResponse{
			ID:      id,
			Created: created,
			Model:   input.Model, // we have to return what the user sent here, due to OpenAI spec.
			Choices: result,
			Object:  "chat.completion",
			Usage: schema.OpenAIUsage{
				PromptTokens:     tokenUsage.Prompt,
				CompletionTokens: tokenUsage.Completion,
				TotalTokens:      tokenUsage.Prompt + tokenUsage.Completion,
			},
		}
		respData, _ := json.Marshal(resp)
		log.Debug().Msgf("Response: %s", respData)

		// Return the prediction in the response body
		return c.JSON(resp)
	}
}

func parseFunctionCall(text string, functionsConfig config.Functions) *FunctionCall {
	actionStartWord := functionsConfig.FunctionStartWord
	actionEndWord := functionsConfig.FunctionEndWord
	actionIgnoreWords := functionsConfig.FunctionIgnoreWords

	useAction := len(actionStartWord) > 0 && len(actionEndWord) > 0
	if !useAction {
		return nil
	}

	startIndex := strings.Index(text, actionStartWord)
	endIndex := strings.LastIndex(text, actionEndWord)

	if startIndex != -1 && endIndex != -1 && startIndex != endIndex {
		startIndex += len(actionStartWord)
		text = text[startIndex:endIndex]
	} else {
		return nil
	}

	for _, word := range actionIgnoreWords {
		text = strings.ReplaceAll(text, word, "")
	}

	functionCall := &FunctionCall{}
	_ = json.Unmarshal([]byte(text), functionCall)
	if functionCall.Arguments == nil {
		functionCall.Arguments = functionCall.Parameters
	}
	if _, ok := functionCall.Arguments.(string); functionCall.Arguments != nil && !ok {
		data, _ := json.Marshal(functionCall.Arguments)
		functionCall.Arguments = string(data)
	}

	return functionCall
}
