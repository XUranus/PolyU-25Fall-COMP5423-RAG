interface SupportModelConfigItem {
    name : string,
    value : string,
    local : boolean // if is a local hugging-face model or a openai API model
}

export const SUPPORT_MODELS : SupportModelConfigItem[] = [
    {
        name : "Qwen/Qwen2.5-0.5B-Instruct",
        value : "Qwen/Qwen2.5-0.5B-Instruct",
        local : true,
    },
    {
        name : "Qwen/qwen-turbo",
        value : "qwen-turbo",
        local : false,
    }
]

export const API_PREFIX : string = "http://127.0.0.1:5000"

export default {
    SUPPORT_MODELS,
    API_PREFIX,
}