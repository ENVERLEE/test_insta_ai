from flask import Flask, render_template, redirect, url_for, flash, request, send_file
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StableDiffusionPipeline
import textwrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from config import Config

class PersonalityForm(FlaskForm):
    question_1 = StringField('1. 나의 가장 행복할 때는?', validators=[DataRequired()])
    question_2 = StringField('2. 너무 슬픈일이 있다. 내 기분은?', validators=[DataRequired()])
    question_3 = StringField('3. 내가 좋아하는 성격은?', validators=[DataRequired()])
    question_4 = StringField('4. 나의 mbti는?', validators=[DataRequired()])
    question_5 = StringField('5. 자신의 성격 중 가장 자랑스러운 점은?', validators=[DataRequired()])
    question_6 = StringField('6. 세상에서 가장 두려운 것은?', validators=[DataRequired()])
    question_7 = StringField('7. 과거로 돌아가서 바꾸고 싶은 일이 있다면?', validators=[DataRequired()])
    question_8 = StringField('8. 가장 평온하고 차분했던 기억은 무엇인가요?', validators=[DataRequired()])
    question_9 = StringField('9. 당신의 삶에서 영감을 준 음악이나 예술 작품은 무엇인가요?', validators=[DataRequired()])
    question_10 = StringField('10. 내가 살고 싶은 세상은?', validators=[DataRequired()])
    submit = SubmitField('제출')

app = Flask(__name__)
app.config.from_object(Config)

# Load models and tokenizers
text_model_name = "Qwen/Qwen2-1.5B-Instruct"
image_model_name = "runwayml/stable-diffusion-v1-5"

tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForCausalLM.from_pretrained(text_model_name).to("cuda" if torch.cuda.is_available() else "cpu")
image_model = StableDiffusionPipeline.from_pretrained(image_model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Path to the TTF font file
FONT_PATH = "./static/fonts/font.ttf"  # Update with the actual path to your TTF file

def load_font_from_path(font_path, font_size):
    """Load a font file from a local path and return a PIL ImageFont object."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Failed to load font.", "error")
        return ImageFont.load_default()
    return font

def wrap_text(text, font, max_width):
    """Wrap text to fit within the specified width."""
    wrapped_lines = textwrap.fill(text, width=max_width, expand_tabs=False, replace_whitespace=False)
    return wrapped_lines.splitlines()

def get_text_size(draw, lines, font):
    """Calculate the size of multi-line text."""
    max_width = 0
    total_height = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_width = max(max_width, line_width)
        total_height += line_height
    return max_width, total_height

def get_dynamic_font(image_size, text):
    """Dynamically determine font size based on image size and text length."""
    width, height = image_size
    font_size = min(width, height) // 10  # Start with a larger font size
    draw = ImageDraw.Draw(Image.new('RGB', (width, height)))
    
    while True:
        font = load_font_from_path(FONT_PATH, font_size)
        wrapped_lines = wrap_text(text, font, width // 10)
        text_width, text_height = get_text_size(draw, wrapped_lines, font)
        if text_width < width - 40 and text_height < height - 40:
            break
        font_size -= 1
        if font_size <= 20:  # Minimum font size
            break
    return font

def generate_image_prompt(poem):
    """Generate a prompt for creating an image without text."""
    return (
        f"Create a visually appealing and artistic image that reflects the mood and themes of the following poem. "
        f"The image should not include any text or writing or similar one. Focus on creating a background that captures the essence "
        f"of the poem's emotions and themes.\n\nPoem:\n{poem}"
    )

@app.route('/', methods=['GET', 'POST'])
def ask_questions():
    form = PersonalityForm()
    if form.validate_on_submit():
        try:
            # Get responses from the form
            questions = [
                form.question_1.data,
                form.question_2.data,
                form.question_3.data,
                form.question_4.data,
                form.question_5.data,
                form.question_6.data,
                form.question_7.data,
                form.question_8.data,
                form.question_9.data,
                form.question_10.data
            ]

            # Create a prompt for the language model
            prompt = (
                "Here are reflections on life:\n"
                f"When I am happiest: {questions[0]}\n"
                f"My feelings when something sad happens: {questions[1]}\n"
                f"The personality traits I admire: {questions[2]}\n"
                f"My MBTI type: {questions[3]}\n"
                f"The aspect of my personality I am most proud of: {questions[4]}\n"
                f"What I fear most: {questions[5]}\n"
                f"If I could change one thing from my past: {questions[6]}\n"
                f"The most peaceful memory I have: {questions[7]}\n"
                f"A piece of art or music that inspired me: {questions[8]}\n"
                f"The world I want to live in: {questions[9]}\n\n"
                "An korean essay on these thoughts:"
            )

            # Tokenize and generate the poem using the local model
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                outputs = text_model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=5,
                    early_stopping=True
                )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(generated_text)
            poem = generated_text

            # Generate an image based on the poem using the local model
            image_prompt = generate_image_prompt(poem)
            image = image_model(image_prompt).images[0]

            font = get_dynamic_font(image.size, poem)
            draw = ImageDraw.Draw(image)

            # Wrap text
            wrapped_lines = wrap_text(poem, font, image.width // 10)
            text_width, text_height = get_text_size(draw, wrapped_lines, font)

            # Center the text
            y = (image.height - text_height) / 2
            for line in wrapped_lines:
                line_width, line_height = draw.textbbox((0, 0), line, font=font)[2:4]
                x = (image.width - line_width) / 2
                draw.text((x, y), line, fill="white", font=font)
                y += line_height + 10  # Add extra space between lines for better readability
            
            img_io = BytesIO()
            image.save(img_io, 'WEBP')
            img_io.seek(0)

            return send_file(img_io, mimetype='image/webp')
        except Exception as e:
            print(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('ask_questions'))
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)